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
#include <time.h>

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600
#define MAX_DEPTH 5
#define SAMPLES_PER_PIXEL 10
#define EPSILON 0.0001f

typedef enum { PLANE, CUBE } ObjectType;

typedef enum { DIFFUSE, METALLIC } MaterialType;

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
} HitInfo;

typedef struct {
    vec3s origin;
    vec3s direction;
} Ray;

#define NUM_OBJECTS 5
Object scene[NUM_OBJECTS];

void setup_scene();
void write_color(uint8_t* pixels, int index, vec3s color);

int rank, size;
vec3s camera_position;
vec3s camera_target;
vec3s camera_up;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    srand(time(NULL) + rank);
    if (rank == 0) {
        printf("Starting Monte Carlo renderer with %d processes\n", size);
    }
    setup_scene();

    camera_position = (vec3s){5.0f, 5.0f, 5.0f};
    camera_target = (vec3s){0.0f, 0.0f, 0.0f};
    camera_up = (vec3s){0.0f, 1.0f, 0.0f};

    if (rank == 0) {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            fprintf(stderr, "SDL could not initialize! SDL_Error: %s\n",
                    SDL_GetError());
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        SDL_Window* window =
            SDL_CreateWindow("MPI Monte Carlo Renderer", SCREEN_WIDTH,
                             SCREEN_HEIGHT, SDL_WINDOW_RESIZABLE);
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

        // Allocate memory for the final image
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

        // Buffer to recieve results from worker
        uint8_t* recv_buffer =
            (uint8_t*)malloc(SCREEN_WIDTH * (SCREEN_HEIGHT / size + 1) * 3);
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
        while (!quit) {
            SDL_Event e;
            while (SDL_PollEvent(&e)) {
                if (e.type == SDL_EVENT_QUIT) {
                    quit = true;
                }
            }
            for (int i = 0; i < size; i++) {
                int start_y = i * (SCREEN_HEIGHT / size);
                int end_y = (i == size - 1) ? SCREEN_HEIGHT
                                            : (i + 1) * (SCREEN_HEIGHT / size);

                if (i > 0) {
                    int params[2] = {start_y, end_y};
                    MPI_Send(params, 2, MPI_INT, i, 0, MPI_COMM_WORLD);
                } else {
                    for (int y = start_y; y < end_y; y++) {
                        for (int x = 0; x < SCREEN_WIDTH; x++) {
                            vec3s color = (vec3s){0.0f, 0.0f, 0.0f};

                            // TODO: path tracing here

                            // Average the samples
                            color.x /= SAMPLES_PER_PIXEL;
                            color.y /= SAMPLES_PER_PIXEL;
                            color.z /= SAMPLES_PER_PIXEL;

                            // Gamma correction
                            color.x = sqrtf(color.x);
                            color.y = sqrtf(color.y);
                            color.z = sqrtf(color.z);

                            // Write to pixel buffer
                            int pixel_index = (y * SCREEN_WIDTH + x) * 3;
                            write_color(pixels, pixel_index, color);
                        }
                    }
                }
            }

            for (int i = 1; i < size; i++) {
                int start_y = i * (SCREEN_HEIGHT / size);
                int end_y = (i == size - 1) ? SCREEN_HEIGHT
                                            : (i + 1) * (SCREEN_HEIGHT / size);
                int chunk_size = (end_y - start_y) * SCREEN_WIDTH * 3;

                MPI_Recv(recv_buffer, chunk_size, MPI_UNSIGNED_CHAR, i, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                memcpy(pixels + start_y * SCREEN_WIDTH * 3, recv_buffer,
                       chunk_size);
            }

            SDL_UpdateTexture(texture, NULL, pixels, SCREEN_WIDTH * 3);
            SDL_RenderClear(renderer);
            SDL_RenderTexture(renderer, texture, NULL, NULL);
            SDL_RenderPresent(renderer);

            // Render once
            SDL_Delay(100);
        }

        // Terminate workers
        for (int i = 1; i < size; i++) {
            int params[2] = {-1, -1};
            MPI_Send(params, 2, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        free(recv_buffer);
        free(pixels);
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    } else {
        uint8_t* worker_pixels = NULL;
        int max_chunk_size = (SCREEN_HEIGHT / size + 1) * SCREEN_WIDTH * 3;
        worker_pixels = (uint8_t*)malloc(max_chunk_size);
        if (!worker_pixels) {
            fprintf(stderr, "Worker %d failed to allocate memory\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        while (1) {
            int params[2];
            MPI_Recv(params, 2, MPI_INT, 0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            int start_y = params[0];
            int end_y = params[1];

            if (start_y == -1) {
                printf("Process %d quitting\n", rank);
                break;
            }

            for (int y = start_y; y < end_y; y++) {
                for (int x = 0; x < SCREEN_WIDTH; x++) {
                    vec3s color = (vec3s){0.0f, 0.0f, 0.0f};

                    // TODO: path tracing here

                    // Average the samples
                    color.x /= SAMPLES_PER_PIXEL;
                    color.y /= SAMPLES_PER_PIXEL;
                    color.z /= SAMPLES_PER_PIXEL;

                    // Gamma correction
                    color.x = sqrtf(color.x);
                    color.y = sqrtf(color.y);
                    color.z = sqrtf(color.z);

                    // Write to worker pixel buffer
                    int local_y = y - start_y;
                    int pixel_index = (local_y * SCREEN_WIDTH + x) * 3;
                    write_color(worker_pixels, pixel_index, color);
                }
            }

            int chunk_size = (end_y - start_y) * SCREEN_WIDTH * 3;
            MPI_Send(worker_pixels, chunk_size, MPI_UNSIGNED_CHAR, 0, 1,
                     MPI_COMM_WORLD);
        }

        free(worker_pixels);
    }
    MPI_Finalize();
    return 0;
}

void setup_scene() {
    // Floor (gray)
    scene[0].type = PLANE;
    scene[0].position = (vec3s){0.0f, -1.0f, 0.0f};
    scene[0].normal = (vec3s){0.0f, 1.0f, 0.0f};
    scene[0].material.type = DIFFUSE;
    scene[0].material.color = (vec3s){0.75f, 0.75f, 0.75f};

    // Red wall
    scene[1].type = PLANE;
    scene[1].position = (vec3s){-3.0f, 0.0f, 0.0f};
    scene[1].normal = (vec3s){1.0f, 0.0f, 0.0f};
    scene[1].material.type = DIFFUSE;
    scene[1].material.color = (vec3s){0.9f, 0.1f, 0.1f};

    // Green wall
    scene[2].type = PLANE;
    scene[2].position = (vec3s){0.0f, 0.0f, -3.0f};
    scene[2].normal = (vec3s){0.0f, 0.0f, 1.0f};
    scene[2].material.type = DIFFUSE;
    scene[2].material.color = (vec3s){0.1f, 0.9f, 0.1f};

    // Metallic cube
    scene[3].type = CUBE;
    scene[3].position = (vec3s){0.0f, 0.0f, 0.0f};
    scene[3].dimensions = (vec3s){1.0f, 1.0f, 1.0f};
    scene[3].material.type = METALLIC;
    scene[3].material.color = (vec3s){0.9f, 0.9f, 0.9f};
    scene[3].material.roughness = 0.05f;

    // Light source
    scene[4].type = CUBE;
    scene[4].position = (vec3s){0.0f, 2.0f, 0.0f};
    scene[4].dimensions = (vec3s){0.5f, 0.1f, 0.5f};
    scene[4].material.type = DIFFUSE;
    scene[4].material.color = (vec3s){5.0f, 5.0f, 5.0f};
}

void write_color(uint8_t* pixels, int index, vec3s color) {
    // clamp
    color.x = fmaxf(0.0f, fminf(1.0f, color.x));
    color.y = fmaxf(0.0f, fminf(1.0f, color.y));
    color.z = fmaxf(0.0f, fminf(1.0f, color.z));

    pixels[index] = (uint8_t)(color.x * 255.0f);
    pixels[index + 1] = (uint8_t)(color.y * 255.0f);
    pixels[index + 2] = (uint8_t)(color.z * 255.0f);
}
