#include <SDL3/SDL.h>
#include <SDL3/SDL_camera.h>
#include <cglm/cglm.h>
#include <cglm/struct.h>
#include <cglm/types-struct.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>  // For memcpy
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600
#define MAX_DEPTH 5
#define SAMPLES_PER_PIXEL 1000
#define EPSILON 0.0001f

typedef enum { PLANE, CUBE } ObjectType;
typedef enum { DIFFUSE, METALLIC, EMISSIVE } MaterialType;

typedef struct {
    MaterialType type;
    vec3s color;
    float roughness;
} Material;

typedef struct {
    ObjectType type;
    vec3s position;
    vec3s dimensions;
    vec3s normal;
    Material material;
} Object;

typedef struct {
    float t;
    vec3s point;
    vec3s normal;
    Object* object;
    bool hit;
    bool front_face;
} HitInfo;

typedef struct {
    vec3s origin;
    vec3s direction;
} Ray;

#define NUM_OBJECTS 5
Object scene[NUM_OBJECTS];

void setup_scene();
void write_color(uint8_t* pixels, int index, vec3s color);

float random_float_0_1();
vec3s trace_path(Ray ray, int depth);
HitInfo intersect_scene(Ray ray);
HitInfo intersect_plane(Ray ray, Object* plane_obj);
HitInfo intersect_cube(Ray ray, Object* cube_obj);
vec3s random_cosine_direction(const vec3s normal);
vec3s reflect(const vec3s incident, const vec3s normal);
void create_orthonormal_basis(const vec3s n, vec3s* b1, vec3s* b2);

int rank_mpi, size_mpi;

vec3s camera_position_global;
mat4s inv_view_matrix_global;
float aspect_ratio_global;
float tan_fov_half_global;

float random_float_0_1() { return (float)rand() / (RAND_MAX + 1.0f); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_mpi);
    MPI_Comm_size(MPI_COMM_WORLD, &size_mpi);
    srand(time(NULL) + rank_mpi);

    if (rank_mpi == 0) {
        printf("Starting Monte Carlo renderer with %d processes\n", size_mpi);
    }
    setup_scene();

    camera_position_global = (vec3s){{5.0f, 5.0f, 5.0f}};
    vec3s camera_target = (vec3s){{0.0f, 0.0f, 0.0f}};
    vec3s camera_up = (vec3s){{0.0f, 1.0f, 0.0f}};

    aspect_ratio_global = (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT;
    float fov_degrees = 60.0f;
    float fov_radians = glm_rad(fov_degrees);
    tan_fov_half_global = tanf(fov_radians / 2.0f);

    mat4s view_matrix =
        glms_lookat(camera_position_global, camera_target, camera_up);
    inv_view_matrix_global = glms_mat4_inv(view_matrix);

    if (rank_mpi == 0) {  // Master
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            fprintf(stderr, "SDL could not initialize! SDL_Error: %s\n",
                    SDL_GetError());
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        SDL_Window* window = SDL_CreateWindow("MPI Monte Carlo Renderer",
                                              SCREEN_WIDTH, SCREEN_HEIGHT, 0);
        if (!window) {
            fprintf(stderr, "Window could not be created! SDL_Error: %s\n",
                    SDL_GetError());
            SDL_Quit();
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        SDL_Renderer* renderer = SDL_CreateRenderer(window, NULL);
        if (!renderer) {
            fprintf(stderr, "Renderer could not be created! SDL_Error: %s\n",
                    SDL_GetError());
            SDL_DestroyWindow(window);
            SDL_Quit();
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        SDL_Texture* texture = SDL_CreateTexture(
            renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING,
            SCREEN_WIDTH, SCREEN_HEIGHT);
        if (!texture) {
            fprintf(stderr, "Texture could not be created! SDL_Error: %s\n",
                    SDL_GetError());
            SDL_DestroyRenderer(renderer);
            SDL_DestroyWindow(window);
            SDL_Quit();
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        uint8_t* pixels = (uint8_t*)malloc(SCREEN_WIDTH * SCREEN_HEIGHT * 3);
        if (!pixels) {
            fprintf(stderr, "Failed to allocate memory for pixels\n");
            SDL_DestroyTexture(texture);
            SDL_DestroyRenderer(renderer);
            SDL_DestroyWindow(window);
            SDL_Quit();
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        int max_rows_per_worker =
            SCREEN_HEIGHT / size_mpi + (SCREEN_HEIGHT % size_mpi > 0 ? 1 : 0);
        uint8_t* recv_buffer =
            (uint8_t*)malloc(SCREEN_WIDTH * max_rows_per_worker * 3);
        if (!recv_buffer) {
            fprintf(stderr, "Failed to allocate memory for receive buffer\n");
            free(pixels);
            SDL_DestroyTexture(texture);
            SDL_DestroyRenderer(renderer);
            SDL_DestroyWindow(window);
            SDL_Quit();
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        bool quit = false;
        double total_render_time_start = MPI_Wtime();
        while (!quit) {
            SDL_Event e;
            while (SDL_PollEvent(&e)) {
                if (e.type == SDL_EVENT_QUIT) {
                    quit = true;
                }
            }
            if (quit) break;

            int rows_per_base_dist = SCREEN_HEIGHT / size_mpi;
            int remainder_rows_dist = SCREEN_HEIGHT % size_mpi;

            for (int i = size_mpi - 1; i >= 0; i--) {
                int start_y =
                    i * rows_per_base_dist +
                    (i < remainder_rows_dist ? i : remainder_rows_dist);
                int end_y = start_y + rows_per_base_dist +
                            (i < remainder_rows_dist ? 1 : 0);

                if (i > 0) {  // Send to workers
                    int params[2] = {start_y, end_y};
                    MPI_Send(params, 2, MPI_INT, i, 0, MPI_COMM_WORLD);
                } else {  // Rank 0 does its part
                    if (end_y - start_y >
                        0) {  // Only print if there are rows to process
                        printf("Rank 0 rendering rows %d to %d...\n", start_y,
                               end_y - 1);
                    }
                    for (int y_coord = start_y; y_coord < end_y; y_coord++) {
                        for (int x_coord = 0; x_coord < SCREEN_WIDTH;
                             x_coord++) {
                            vec3s accumulated_color = glms_vec3_zero();
                            for (int s = 0; s < SAMPLES_PER_PIXEL; s++) {
                                float Px_ndc =
                                    (2.0f *
                                     ((float)x_coord + random_float_0_1()) /
                                     SCREEN_WIDTH) -
                                    1.0f;
                                float Py_ndc =
                                    1.0f -
                                    (2.0f *
                                     ((float)y_coord + random_float_0_1()) /
                                     SCREEN_HEIGHT);
                                float Px_cam = Px_ndc * aspect_ratio_global *
                                               tan_fov_half_global;
                                float Py_cam = Py_ndc * tan_fov_half_global;

                                vec4s ray_dir_camera_s =
                                    (vec4s){{Px_cam, Py_cam, -1.0f, 0.0f}};
                                vec4s ray_dir_world_s = glms_mat4_mulv(
                                    inv_view_matrix_global, ray_dir_camera_s);
                                Ray current_ray;
                                current_ray.origin = camera_position_global;
                                current_ray.direction = glms_normalize((vec3s){
                                    {ray_dir_world_s.x, ray_dir_world_s.y,
                                     ray_dir_world_s.z}});

                                vec3s sample_color =
                                    trace_path(current_ray, MAX_DEPTH);
                                accumulated_color = glms_vec3_add(
                                    accumulated_color, sample_color);
                            }
                            vec3s final_color = glms_vec3_scale(
                                accumulated_color, 1.0f / SAMPLES_PER_PIXEL);

                            // Protected Gamma correction
                            final_color.x = sqrtf(fmaxf(0.0f, final_color.x));
                            final_color.y = sqrtf(fmaxf(0.0f, final_color.y));
                            final_color.z = sqrtf(fmaxf(0.0f, final_color.z));

                            int pixel_index =
                                (y_coord * SCREEN_WIDTH + x_coord) * 3;
                            write_color(pixels, pixel_index, final_color);
                        }
                    }
                    if (end_y - start_y > 0) {
                        printf("Rank 0 finished its rows.\n");
                    }
                }
            }

            int rows_per_base_recv = SCREEN_HEIGHT / size_mpi;
            int remainder_rows_recv = SCREEN_HEIGHT % size_mpi;

            for (int i = 1; i < size_mpi; i++) {
                int start_y_recv =
                    i * rows_per_base_recv +
                    (i < remainder_rows_recv ? i : remainder_rows_recv);
                int end_y_recv = start_y_recv + rows_per_base_recv +
                                 (i < remainder_rows_recv ? 1 : 0);
                int num_rows_for_worker_i = end_y_recv - start_y_recv;

                if (num_rows_for_worker_i > 0) {
                    int chunk_size = num_rows_for_worker_i * SCREEN_WIDTH * 3;
                    MPI_Recv(recv_buffer, chunk_size, MPI_UNSIGNED_CHAR, i, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    memcpy(pixels + start_y_recv * SCREEN_WIDTH * 3,
                           recv_buffer, chunk_size);
                }
            }

            SDL_UpdateTexture(texture, NULL, pixels, SCREEN_WIDTH * 3);
            SDL_RenderClear(renderer);
            SDL_RenderTexture(renderer, texture, NULL, NULL);
            SDL_RenderPresent(renderer);

            double total_render_time_end = MPI_Wtime();
            printf(
                "Frame processed. Total time: %.2f seconds. Waiting for quit "
                "signal or next frame event.\n",
                total_render_time_end - total_render_time_start);
            total_render_time_start = total_render_time_end;

            // One frame generation only
            if (!quit) {
                printf("Render complete. Waiting for quit signal.\n");
                while (!quit) {
                    while (SDL_PollEvent(&e)) {
                        if (e.type == SDL_EVENT_QUIT) {
                            quit = true;
                        }
                    }
                    if (quit) break;
                    SDL_Delay(100);
                }
            }
        }

        for (int i = 1; i < size_mpi; i++) {
            int params[2] = {-1, -1};
            MPI_Send(params, 2, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        free(recv_buffer);
        free(pixels);
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();

    } else {  // Worker processes
        int max_rows_for_this_worker =
            SCREEN_HEIGHT / size_mpi + (SCREEN_HEIGHT % size_mpi > 0 ? 1 : 0);
        uint8_t* worker_pixels =
            (uint8_t*)malloc(SCREEN_WIDTH * max_rows_for_this_worker * 3);
        if (!worker_pixels) {
            fprintf(stderr, "Worker %d failed to allocate memory\n", rank_mpi);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        while (1) {
            int params[2];
            MPI_Recv(params, 2, MPI_INT, 0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            int start_y = params[0];
            int end_y = params[1];

            if (start_y == -1) {  // Termination signal
                printf("Process %d quitting\n", rank_mpi);
                break;
            }

            int local_row_idx = 0;
            for (int y_coord = start_y; y_coord < end_y; y_coord++) {
                for (int x_coord = 0; x_coord < SCREEN_WIDTH; x_coord++) {
                    vec3s accumulated_color = glms_vec3_zero();
                    for (int s = 0; s < SAMPLES_PER_PIXEL; s++) {
                        float Px_ndc =
                            (2.0f * ((float)x_coord + random_float_0_1()) /
                             SCREEN_WIDTH) -
                            1.0f;
                        float Py_ndc =
                            1.0f -
                            (2.0f * ((float)y_coord + random_float_0_1()) /
                             SCREEN_HEIGHT);
                        float Px_cam =
                            Px_ndc * aspect_ratio_global * tan_fov_half_global;
                        float Py_cam = Py_ndc * tan_fov_half_global;

                        vec4s ray_dir_camera_s =
                            (vec4s){{Px_cam, Py_cam, -1.0f, 0.0f}};
                        vec4s ray_dir_world_s = glms_mat4_mulv(
                            inv_view_matrix_global, ray_dir_camera_s);
                        Ray current_ray;
                        current_ray.origin = camera_position_global;
                        current_ray.direction = glms_normalize(
                            (vec3s){{ray_dir_world_s.x, ray_dir_world_s.y,
                                     ray_dir_world_s.z}});

                        vec3s sample_color = trace_path(current_ray, MAX_DEPTH);
                        accumulated_color =
                            glms_vec3_add(accumulated_color, sample_color);
                    }
                    vec3s final_color = glms_vec3_scale(
                        accumulated_color, 1.0f / SAMPLES_PER_PIXEL);

                    // Protected Gamma correction
                    final_color.x = sqrtf(fmaxf(0.0f, final_color.x));
                    final_color.y = sqrtf(fmaxf(0.0f, final_color.y));
                    final_color.z = sqrtf(fmaxf(0.0f, final_color.z));

                    int pixel_index =
                        (local_row_idx * SCREEN_WIDTH + x_coord) * 3;
                    write_color(worker_pixels, pixel_index, final_color);
                }
                local_row_idx++;
            }
            int num_rows_processed = end_y - start_y;
            if (num_rows_processed > 0) {
                int chunk_size = num_rows_processed * SCREEN_WIDTH * 3;
                MPI_Send(worker_pixels, chunk_size, MPI_UNSIGNED_CHAR, 0, 1,
                         MPI_COMM_WORLD);
            }
        }
        free(worker_pixels);
    }
    MPI_Finalize();
    return 0;
}

void setup_scene() {
    scene[0] = (Object){PLANE,
                        (vec3s){{0.0f, -1.0f, 0.0f}},
                        GLMS_VEC3_ZERO,
                        (vec3s){{0.0f, 1.0f, 0.0f}},
                        {DIFFUSE, (vec3s){{0.75f, 0.75f, 0.75f}}, 0.0f}};

    scene[1] = (Object){PLANE,
                        (vec3s){{-3.0f, 0.0f, 0.0f}},
                        GLMS_VEC3_ZERO,
                        (vec3s){{1.0f, 0.0f, 0.0f}},
                        {DIFFUSE, (vec3s){{0.9f, 0.1f, 0.1f}}, 0.0f}};

    scene[2] = (Object){PLANE,
                        (vec3s){{0.0f, 0.0f, -3.0f}},
                        GLMS_VEC3_ZERO,
                        (vec3s){{0.0f, 0.0f, 1.0f}},
                        {DIFFUSE, (vec3s){{0.1f, 0.9f, 0.1f}}, 0.0f}};

    scene[3] = (Object){CUBE,
                        (vec3s){{1.0f, 0.0f, 1.0f}},
                        (vec3s){{2.0f, 2.0f, 2.0f}},
                        GLMS_VEC3_ZERO,
                        {METALLIC, (vec3s){{0.9f, 0.9f, 0.9f}}, 0.05f}};

    scene[4] = (Object){CUBE,
                        (vec3s){{0.0f, 10.0f, 0.0f}},
                        (vec3s){{3.0f, 0.1f, 3.0f}},
                        GLMS_VEC3_ZERO,
                        {EMISSIVE, (vec3s){{15.0f, 15.0f, 12.0f}}, 0.0f}};
}

void write_color(uint8_t* pixels, int index, vec3s color) {
    pixels[index] = (uint8_t)(fmaxf(0.0f, fminf(1.0f, color.x)) * 255.999f);
    pixels[index + 1] = (uint8_t)(fmaxf(0.0f, fminf(1.0f, color.y)) * 255.999f);
    pixels[index + 2] = (uint8_t)(fmaxf(0.0f, fminf(1.0f, color.z)) * 255.999f);
}

// --- Intersection and Path Tracing Logic ---

HitInfo intersect_plane(Ray ray, Object* plane_obj) {
    HitInfo hit_info;
    hit_info.hit = false;
    hit_info.object = plane_obj;

    float denom = glms_vec3_dot(plane_obj->normal, ray.direction);
    if (fabsf(denom) > EPSILON) {
        vec3s p0_l0 = glms_vec3_sub(plane_obj->position, ray.origin);
        float t = glms_vec3_dot(p0_l0, plane_obj->normal) / denom;
        if (t >= EPSILON) {
            hit_info.t = t;
            hit_info.point =
                glms_vec3_add(ray.origin, glms_vec3_scale(ray.direction, t));
            hit_info.normal = plane_obj->normal;  // Initial normal
            hit_info.front_face =
                glms_vec3_dot(ray.direction, plane_obj->normal) < 0;
            if (!hit_info.front_face) {
                hit_info.normal = glms_vec3_scale(
                    hit_info.normal, -1.0f);  // Normal points against ray
            }
            hit_info.hit = true;
        }
    }
    return hit_info;
}

HitInfo intersect_cube(Ray ray, Object* cube_obj) {  // AABB Intersection
    HitInfo hit_info;
    hit_info.hit = false;
    hit_info.object = cube_obj;

    vec3s half_dims = glms_vec3_scale(cube_obj->dimensions, 0.5f);
    vec3s min_b = glms_vec3_sub(cube_obj->position, half_dims);
    vec3s max_b = glms_vec3_add(cube_obj->position, half_dims);

    float t_min = -INFINITY, t_max = INFINITY;
    int hit_axis[2] = {-1,
                       -1};  // To store axis of intersection for normal calc

    for (int i = 0; i < 3; i++) {  // x, y, z axes
        float inv_dir = 1.0f / ray.direction.raw[i];
        float t0 = (min_b.raw[i] - ray.origin.raw[i]) * inv_dir;
        float t1 = (max_b.raw[i] - ray.origin.raw[i]) * inv_dir;
        if (inv_dir < 0.0f) {
            float temp = t0;
            t0 = t1;
            t1 = temp;
        }

        if (t0 > t_min) {
            t_min = t0;
            hit_axis[0] = i;
        }
        if (t1 < t_max) {
            t_max = t1;
            hit_axis[1] = i;
        }  // Not strictly needed for normal if only t_min is used

        if (t_min > t_max) return hit_info;  // No intersection
    }

    if (t_min >= EPSILON &&
        t_min < INFINITY) {  // Check t_min as the closer hit
        hit_info.t = t_min;
        hit_info.point =
            glms_vec3_add(ray.origin, glms_vec3_scale(ray.direction, t_min));
        hit_info.hit = true;

        // Calculate normal
        vec3s p_local = glms_vec3_sub(hit_info.point, cube_obj->position);
        hit_info.normal = glms_vec3_zero();
        float max_dist = -1.0f;  // Find which face was hit by checking distance
                                 // to center plane of face
        int best_axis = -1;

        for (int i = 0; i < 3; ++i) {
            float dist_to_face_center = fabsf(p_local.raw[i]);
            if (fabsf(dist_to_face_center - half_dims.raw[i]) <
                EPSILON * 10.0f) {  // If close to face i
                best_axis = i;
                break;
            }
        }
        if (best_axis != -1) {
            hit_info.normal.raw[best_axis] =
                (p_local.raw[best_axis] > 0) ? 1.0f : -1.0f;
        } else {  // Fallback if point is not exactly on a face (should be rare
                  // with AABB)
            // This needs robust handling; for now, use the axis associated with
            // t_min if possible This normal calculation for AABB can be tricky.
            // A common way:
            vec3s c = glms_vec3_sub(hit_info.point, cube_obj->position);
            vec3s d = glms_vec3_divs(
                c, half_dims.x);  // Assuming uniform half_dims for simplicity
                                  // here, else component-wise
            float max_comp = 0.0;
            int normal_idx = 0;
            for (int i = 0; i < 3; i++) {
                if (fabsf(d.raw[i]) > max_comp) {
                    max_comp = fabsf(d.raw[i]);
                    normal_idx = i;
                }
            }
            hit_info.normal = glms_vec3_zero();
            hit_info.normal.raw[normal_idx] =
                (d.raw[normal_idx] > 0.0) ? 1.0 : -1.0;
        }

        hit_info.front_face = glms_vec3_dot(ray.direction, hit_info.normal) < 0;
        // AABBs are typically solid, so if ray originates inside, this normal
        // logic needs care For simplicity, we assume rays start outside.
    }
    return hit_info;
}

HitInfo intersect_scene(Ray ray) {
    HitInfo closest_hit;
    closest_hit.hit = false;
    closest_hit.t = INFINITY;

    for (int i = 0; i < NUM_OBJECTS; ++i) {
        HitInfo current_hit;
        if (scene[i].type == PLANE) {
            current_hit = intersect_plane(ray, &scene[i]);
        } else if (scene[i].type == CUBE) {
            current_hit = intersect_cube(ray, &scene[i]);
        } else {
            current_hit.hit = false;
        }

        if (current_hit.hit && current_hit.t < closest_hit.t) {
            closest_hit = current_hit;
        }
    }
    return closest_hit;
}

// Helper to create an orthonormal basis (tangent and bitangent) from a normal
// vector
void create_orthonormal_basis(const vec3s n, vec3s* b1, vec3s* b2) {
    if (fabsf(n.x) > fabsf(n.y)) {
        *b1 = (vec3s){{-n.z, 0.0f, n.x}};
    } else {
        *b1 = (vec3s){{0.0f, n.z, -n.y}};
    }
    *b1 = glms_vec3_normalize(*b1);
    *b2 = glms_vec3_cross(n, *b1);
    // *b2 will already be normalized if n and *b1 are unit and orthogonal
}

// Generates a random direction on the hemisphere oriented along normal,
// sampled with a cosine distribution.
vec3s random_cosine_direction(const vec3s normal) {
    float r1 = random_float_0_1();  // For sqrt(1-r2*r2) effectively for z'
    float r2 = random_float_0_1();  // For phi

    float sin_theta = sqrtf(
        1.0f -
        r1 * r1);  // sin_theta = sqrt(1 - cos^2(theta_prime)) but
                   // cos^2(theta_prime) is r1 for this sampling More directly:
                   // r1 is actually cos(theta_prime), so sin_theta_prime =
                   // sqrt(1-r1^2) NO Hanrahan p.36: z' = sqrt(ξ1), so
                   // cos(theta') = sqrt(ξ1) Using Malley's method (cosine
                   // weighted): x' = cos(2πξ2)sqrt(ξ1) y' = sin(2πξ2)sqrt(ξ1)
                   // z' = sqrt(1-ξ1)
                   // This samples points on a disk then projects to hemisphere
    float z_local =
        sqrtf(r1);  // cos(theta_prime) where theta_prime is angle with sampled
                    // vector and local Z axis of hemisphere
    float x_local = cosf(2.0f * (float)M_PI * r2) * sqrtf(1.0f - r1);
    float y_local = sinf(2.0f * (float)M_PI * r2) * sqrtf(1.0f - r1);

    vec3s tangent, bitangent;
    create_orthonormal_basis(normal, &tangent, &bitangent);

    vec3s world_dir = glms_vec3_zero();
    world_dir = glms_vec3_add(world_dir, glms_vec3_scale(tangent, x_local));
    world_dir = glms_vec3_add(world_dir, glms_vec3_scale(bitangent, y_local));
    world_dir = glms_vec3_add(world_dir, glms_vec3_scale(normal, z_local));

    return glms_vec3_normalize(
        world_dir);  // Should already be normalized if basis is orthonormal
}

vec3s reflect(const vec3s incident, const vec3s normal) {
    return glms_vec3_sub(
        incident,
        glms_vec3_scale(normal, 2.0f * glms_vec3_dot(incident, normal)));
}

vec3s trace_path(Ray ray, int depth) {
    if (depth <= 0) {
        return glms_vec3_zero();  // Max depth reached
    }

    HitInfo hit = intersect_scene(ray);

    if (!hit.hit) {  // Ray missed all objects, return background/environment
                     // color
        // Simple gradient sky
        // float t = 0.5f * (ray.direction.y + 1.0f);
        // vec3s sky_bottom = (vec3s){{1.0f, 1.0f, 1.0f}};
        // vec3s sky_top    = (vec3s){{0.5f, 0.7f, 1.0f}};
        // return glms_vec3_lerp(sky_bottom, sky_top, t);
        return glms_vec3_zero();  // Black background
    }

    Material mat = hit.object->material;
    vec3s emitted = glms_vec3_zero();
    if (mat.type == EMISSIVE) {
        // For path tracing, emissive surfaces usually only contribute if hit
        // directly by camera ray or if explicitly sampled (Next Event
        // Estimation). If a path bounces and *then* hits an emissive surface,
        // that counts as indirect light. For a simple path tracer, if a ray
        // from camera hits light, it's Le. If a ray from a surface hits light,
        // it's Li.
        return mat.color;
    }

    vec3s new_ray_dir;
    vec3s albedo = mat.color;  // Default to material color as albedo

    if (mat.type == DIFFUSE) {
        new_ray_dir = random_cosine_direction(hit.normal);
    } else if (mat.type == METALLIC) {
        new_ray_dir = reflect(ray.direction, hit.normal);
        // Simplistic roughness: mix perfect reflection with diffuse
        if (mat.roughness > 0.0f) {
            vec3s diffuse_dir = random_cosine_direction(hit.normal);
            new_ray_dir =
                glms_vec3_lerp(new_ray_dir, diffuse_dir, mat.roughness);
            new_ray_dir = glms_vec3_normalize(new_ray_dir);
        }
    } else {
        return glms_vec3_zero();  // Should not happen if EMISSIVE is handled
                                  // above
    }

    Ray new_ray;
    new_ray.origin = glms_vec3_add(
        hit.point,
        glms_vec3_scale(hit.normal,
                        EPSILON * 2.0f));  // Offset to avoid self-intersection
    new_ray.direction = new_ray_dir;

    // Recursive call for the bounced ray
    vec3s incoming_light = trace_path(new_ray, depth - 1);

    // For diffuse, the BRDF * cos(theta) / PDF simplifies to albedo
    // For perfect metallic, same idea.
    // This is a simplification that works for these ideal materials.
    return glms_vec3_mul(albedo, incoming_light);
}
