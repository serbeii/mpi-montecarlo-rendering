Depends on cglm, sdl3 and an mpi implementation
To compile the program use:
```
mpicc main.c -o main.out $(pkg-config --cflags --libs sdl3) -lm
```
