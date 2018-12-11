
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "mpi.h"

#define N_DIMENSION 2
#define MASTER 0
// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define TOP_LEFT = 1
#define BOTTOM_LEFT = 0

void top_left_corner(const int nx, const int ny, float * restrict image, float * restrict tmp_image);
void bottom_left_corner(const int nx, const int ny, float * restrict image, float * restrict tmp_image);
void top(const int nx, const int ny, float * restrict image, float * restrict tmp_image);
void bottom(const int nx, const int ny, float * restrict image, float * restrict tmp_image);
void top_right_corner(const int nx, const int ny, float * restrict image, float * restrict tmp_image);
void bottom_right_corner(const int nx, const int ny, float * restrict image, float * restrict tmp_image);
void stencil(const int nx, const int ny, float * restrict image, float * restrict tmp_image);
void init_image(const int nx, const int ny, float * restrict image, float * restrict tmp_image);
void output_image(const char * file_name, const int nx, const int ny, float * restrict image);
int calc_nrows_from_rank(int rank, int size, int rows);
int calc_ncols_from_rank(int rank, int size, int cols);
double wtime(void);

int main(int argc, char *argv[]) {
  int ii;                /* generic counter */
  int rank;            /* the rank of this process */
  int size;              /* number of processes in the communicator */
  int direction;         /* the coordinate dimension of a shift */
  int disp;              /* displacement, >1 is 'forwards', <1 is 'backwards' along a dimension */
  int dest;
  int source;
  double *sendbuf;       /* buffer to hold values to send */
  double *recvbuf;       /* buffer to hold received values */
  float *image_pad;
  float *tmp_image_pad;
  int tag = 0;
  MPI_Status status;

  int loop_row_start_point;
  int loop_col_start_point;
  int loop_row_end_point;
  int loop_col_end_point;

  int north;             /* the rank of the process above this rank in the grid */
  int south;             /* the rank of the process below this rank in the grid */
  int east;              /* the rank of the process to the right of this rank in the grid */
  int west;              /* the rank of the process to the left of this rank in the grid */


  int reorder = 0;       /* an argument to MPI_Cart_create() */
  int dims[N_DIMENSION];       /* array to hold dimensions of an N_DIMENSION grid of processes */
  int periods[N_DIMENSION];    /* array to specificy periodic boundary conditions on each dimension */
  int coords[N_DIMENSION];     /* array to hold the grid coordinates for a rank */
  MPI_Comm comm_cart;    /* a cartesian topology aware communicator */

  int bottom_left;
  int bottom_right;


  int local_nrows;
  int local_ncols;
  int local_usual_ncols;
  int local_usual_nrows;
  int remote_ncols;
  // initialise our MPI environment
  MPI_Init( &argc, &argv);

  // check wheter the initialisation was successful

  // determine the size of the group of processes associated with the 'communicator'.
  // default communicator is MPI_COMM_WORLD, consisting of all the processes in the launched MPI 'job'
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  if (N_DIMENSION != 2) {
    fprintf(stderr, "Error: number of dimension is assumed to be 2\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  if (size < (N_DIMENSION * N_DIMENSION)) {
    fprintf(stderr,"Error: size assumed to be at least N_DIMENSION * N_DIMENSION, i.e. 4.\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  if ((size % 2) > 0) {
    fprintf(stderr,"Error: size assumed to be even.\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  for (ii=0; ii<N_DIMENSION; ii++) {
    dims[ii] = 0;
    periods[ii] = 1; /* set periodic boundary conditions to True for all dimensions */
  }

  MPI_Dims_create(size, N_DIMENSION, dims);
  if(rank == MASTER) {
    printf("ranks spread over a grid of %d dimension(s): [%d,%d]\n", N_DIMENSION, dims[0], dims[1]);
  }

  MPI_Cart_create(MPI_COMM_WORLD, N_DIMENSION, dims, periods, reorder, &comm_cart);

  MPI_Cart_coords(comm_cart, rank, N_DIMENSION, coords);
  MPI_Barrier(MPI_COMM_WORLD);
  printf("rank %d has coordinates (%d,%d)\n", rank, coords[0], coords[1]);

  direction = 0;
  disp = 1;
  MPI_Cart_shift(comm_cart, direction, disp, &west, &east);
  direction = 1;
  disp = 1;
  MPI_Cart_shift(comm_cart, direction, disp, &south, &north);

  MPI_Barrier(MPI_COMM_WORLD);
  printf("rank: %d\n\tnorth=%d\n\tsouth=%d\n\teast=%d\n\twest=%d\n", rank,north,south,east,west);

  bottom_left = size - 2;
  bottom_right = size - 1;


  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  local_nrows = calc_nrows_from_rank(rank, size, nx);
  local_ncols = calc_ncols_from_rank(rank, size, ny);

  printf("Rank: %d, rows: %d, colmns %d\n", rank, local_nrows, local_ncols);

  float * restrict image_original = malloc(sizeof(float) * nx * ny);
  float * restrict tmp_image_original = malloc(sizeof(float) * nx * ny);

  init_image(nx, ny, image_original, tmp_image_original);

  local_usual_ncols = ny / (size * 0.5);
  local_usual_nrows = nx / 2;

  if ((rank % 2) == 1) {
    loop_row_start_point = 0;
    loop_row_end_point = local_usual_nrows;
  } else {
    loop_row_start_point = local_usual_nrows;
    loop_row_end_point = nx;
  }

  loop_col_start_point = (rank / 2) * local_usual_ncols;
  loop_col_end_point = loop_col_start_point + local_ncols;


  float * restrict image = malloc(sizeof(float) * local_nrows * local_ncols);
  float * restrict tmp_image = malloc(sizeof(float) * local_nrows * local_ncols);

  if (rank == MASTER || rank == 1 || rank == bottom_left || rank == bottom_right) {
    image_pad = (float*)malloc(sizeof(float) * (local_nrows + 1) * (local_ncols + 1));
    tmp_image_pad = (float*)malloc(sizeof(float) * (local_nrows + 1) * (local_ncols + 1));
  } else {
    image_pad = (float*)malloc(sizeof(float) * (local_nrows + 1) * (local_ncols + 2));
    tmp_image_pad = (float*)malloc(sizeof(float) * (local_nrows + 1) * (local_ncols + 2));
  }

  for (int i = loop_row_start_point; i < loop_row_end_point; i++) {
    for (int j = loop_col_start_point; j < loop_col_end_point; j++) {
      image[(j - loop_col_start_point) + (i - loop_row_start_point) * local_ncols] = image_original[j + i * ny];
      tmp_image[(j - loop_col_start_point) + (i - loop_row_start_point) * local_ncols] = tmp_image_original[j + i * ny];
    }
  }

  sendbuf = (double*)malloc(sizeof(double) * local_nrows);
  recvbuf = (double*)malloc(sizeof(double) * local_nrows);

  /* The last rank has the most columns apportioned.
     printbuf must be big enough to hold this number */
  remote_ncols = calc_ncols_from_rank(size-1, size, ny);

  double tic = wtime();

  if (rank == MASTER) {

    for (int j = 0; j < local_ncols; j++) {
      sendbuf[j] = image[(local_ncols - 1) + j * local_nrows];
    }

    MPI_Sendrecv(sendbuf, local_ncols, MPI_DOUBLE, north, tag, recvbuf, local_ncols, MPI_DOUBLE, north, tag, MPI_COMM_WORLD, &status);

    for (int j = 0; j < local_ncols; j++) {
      image_pad[local_nrows + j * (local_nrows + 1)] = recvbuf[j];
    }

    for (int i = 1; i < local_nrows + 1; i++) {
      for (int j = 0; j < local_ncols; j++) {
        image_pad[j + i * local_ncols] = image[j + (i - 1) * local_ncols];
        tmp_image_pad[j + i * local_ncols] = image[j + (i - 1) * local_ncols];
      }
    }

    for (int t = 0; t < niters; t++) {
      top_left_corner(local_nrows, local_ncols, image_pad, tmp_image_pad);
      top_left_corner(local_nrows, local_ncols, tmp_image_pad, image_pad);
    }
  } else {
    if (rank == 1) {

      for (int j = 0; j < local_ncols; j++) {
        sendbuf[j] = image[j * local_nrows];
      }

      MPI_Sendrecv(sendbuf, local_ncols, MPI_DOUBLE, south, tag, recvbuf, local_ncols, MPI_DOUBLE, south, tag, MPI_COMM_WORLD, &status);

      for (int j = 0; j < local_ncols; j++) {
        image_pad[j * (local_nrows + 1)] = recvbuf[j];
      }

      for (int i = 0; i < local_nrows; i++) {
        for (int j = 0; j < local_ncols; j++) {
          image_pad[j + i * local_ncols] = image[j + i * local_ncols];
          tmp_image_pad[j + i * local_ncols] = image[j + i * local_ncols];
        }
      }

      for (int t = 0; t < niters; t++) {
        top_right_corner(local_nrows, local_ncols, image_pad, tmp_image_pad);
        top_right_corner(local_nrows, local_ncols, tmp_image_pad, image_pad);
      }
    } else {
      if (rank == bottom_left) {

        for (int i = 0; i < local_nrows; i++) {
          for (int j = 1; j < local_ncols + 1; j++) {
            image_pad[j + i * local_ncols] = image[(j - 1) + i * local_ncols];
            tmp_image_pad[j + i * local_ncols] = image[(j - 1) + i * local_ncols];
          }
        }


        for (int t = 0; t < niters; t++) {
          bottom_left_corner(local_nrows, local_ncols, image_pad, tmp_image_pad);
          bottom_left_corner(local_nrows, local_ncols, tmp_image_pad, image_pad);
        }
      } else {
        if (rank == bottom_right) {

          for (int i = 1; i < local_nrows + 1; i++) {
            for (int j = 1; j < local_ncols + 1; j++) {
              image_pad[j + i * local_ncols] = image[j + (i - 1) * local_ncols];
              tmp_image_pad[j + i * local_ncols] = image[j + (i - 1) * local_ncols];
            }
          }

          for (int t = 0; t < niters; t++) {
            bottom_right_corner(local_nrows, local_ncols, image_pad, tmp_image_pad);
            bottom_right_corner(local_nrows, local_ncols, tmp_image_pad, image_pad);
          }
    //     } else {
    //       if ((rank % 2) == 1) {
    //         for (int t = 0; t < niters; t++) {
    //           top(local_nrows, local_ncols, image_pad, tmp_image_pad);
    //           top(local_nrows, local_ncols, tmp_image_pad, image_pad);
    //         }
    //       } else {
    //         for (int t = 0; t < niters; t++) {
    //           bottom(local_nrows, local_ncols, image_pad, tmp_image_pad);
    //           bottom(local_nrows, local_ncols, tmp_image_pad, image_pad);
    //         }
    //       }
        }
      }
    }
  }

  double toc = wtime();


  // Allocate the image
  // float * restrict image = malloc(sizeof(float)*nx*ny);
  // float * restrict tmp_image = malloc(sizeof(float)*nx*ny);
  //
  //
  // // Set the input image
  // init_image(nx, ny, image, tmp_image);
  //
  // // Call the stencil kernel
  // double tic = wtime();
  //
  // for (int t = 0; t < niters; ++t) {
  //   stencil(nx, ny, image, tmp_image);
  //   stencil(nx, ny, tmp_image, image);
  // }
  // double toc = wtime();


  // Output
  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc-tic);
  printf("------------------------------------\n");


  if (rank == 2){
    output_image("RANK2.pgm", local_nrows, local_ncols, image_pad);
  }

  free(image);
  free(tmp_image);
  free(sendbuf);
  free(recvbuf);

  MPI_Finalize();

  return EXIT_SUCCESS;
}

void top_left_corner(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  float initialMul = 0.6f;
  float Mul = 0.1f;
  float numberToadd = 0.0f;

  // when i = 0, j = 0

  numberToadd = image[0] * initialMul;
  numberToadd += image[ny] * Mul;
  numberToadd += image[1] * Mul;
  tmp_image[0] = numberToadd;

  // when i = 0, 0 < j < ny - 1
  for (int j = 1; j < ny - 1; j++) {
    numberToadd = image[j] * initialMul;
    numberToadd += image[j + ny] * Mul;
    numberToadd += image[j-1] * Mul;
    numberToadd += image[j+1] * Mul;
    tmp_image[j] = numberToadd;

  }

  // when 0 < i < nx -1, j = 0
  for (int i = 1; i < nx - 1; i++) {
    numberToadd = image[i * ny] * initialMul;
    numberToadd += image[(i-1) * ny] * Mul;
    numberToadd += image[(i+1) * ny] * Mul;
    numberToadd += image[1 + i * ny] * Mul;
    tmp_image[i * ny] = numberToadd;
  }

  for (int i = 1; i < nx - 1; i++) {
    for (int j = 1; j < ny - 1; j++) {
      tmp_image[j+i*ny] = image[j+i*ny] * initialMul;
      tmp_image[j+i*ny] += image[j  +(i-1)*ny] * Mul;
      tmp_image[j+i*ny] += image[j  +(i+1)*ny] * Mul;
      tmp_image[j+i*ny] += image[j-1+i*ny] * Mul;
      tmp_image[j+i*ny] += image[j+1+i*ny] * Mul;

      // tmp_image[j+i*ny] = numberToadd;
    }
  }



}

void bottom_left_corner(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  float initialMul = 0.6f;
  float Mul = 0.1f;
  float numberToadd = 0.0f;

  // when i = 0, j = ny - 1
  numberToadd = image[(ny - 1)] * initialMul;
  numberToadd += image[(ny - 1) + ny] * Mul;
  numberToadd += image[(ny - 2)] * Mul;
  tmp_image[(ny - 1)] = numberToadd;

  // when i = 0, 0 < j < ny - 1
  for (int j = 1; j < ny - 1; j++) {
  	numberToadd = image[j] * initialMul;
  	numberToadd += image[j + ny] * Mul;
  	numberToadd += image[j-1] * Mul;
  	numberToadd += image[j+1] * Mul;
  	tmp_image[j] = numberToadd;
  }

  // when 0 < i < nx - 1, j = ny - 1
  for (int i = 1; i < nx - 1; i++) {
    numberToadd = image[(ny - 1) + i * ny] * initialMul;
  	numberToadd += image[(ny - 1) + (i - 1) * ny] * Mul;
  	numberToadd += image[(ny - 1) + (i + 1) * ny] * Mul;
  	numberToadd += image[(ny - 2) + i * ny] * Mul;
  	tmp_image[(ny - 1) + i * ny] = numberToadd;
  }

  for (int i = 1; i < nx - 1; i++) {
    for (int j = 1; j < ny - 1; j++) {
      tmp_image[j+i*ny] = image[j+i*ny] * initialMul;
      tmp_image[j+i*ny] += image[j  +(i-1)*ny] * Mul;
      tmp_image[j+i*ny] += image[j  +(i+1)*ny] * Mul;
      tmp_image[j+i*ny] += image[j-1+i*ny] * Mul;
      tmp_image[j+i*ny] += image[j+1+i*ny] * Mul;

      // tmp_image[j+i*ny] = numberToadd;
    }
  }

}

void left(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  float initialMul = 0.6f;
  float Mul = 0.1f;
  float numberToadd = 0.0f;

  // when i = 0, 0 < j < ny - 1
  for (int j = 1; j < ny - 1; j++) {
  	numberToadd = image[j] * initialMul;
  	numberToadd += image[j + ny] * Mul;
  	numberToadd += image[j-1] * Mul;
  	numberToadd += image[j+1] * Mul;
  	tmp_image[j] = numberToadd;
  }

  for (int i = 1; i < nx - 1; i++) {
    for (int j = 1; j < ny - 1; j++) {
      tmp_image[j+i*ny] = image[j+i*ny] * initialMul;
      tmp_image[j+i*ny] += image[j  +(i-1)*ny] * Mul;
      tmp_image[j+i*ny] += image[j  +(i+1)*ny] * Mul;
      tmp_image[j+i*ny] += image[j-1+i*ny] * Mul;
      tmp_image[j+i*ny] += image[j+1+i*ny] * Mul;

      // tmp_image[j+i*ny] = numberToadd;
    }
  }
}

void right(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  float initialMul = 0.6f;
  float Mul = 0.1f;
  float numberToadd = 0.0f;

  // when i = nx - 1, 0 < j < ny - 1

  for (int j = 1; j < ny - 1; j++) {
    numberToadd = image[j+(nx-1)*ny] * initialMul;
    numberToadd += image[j+(nx-2)*ny] * Mul;
    numberToadd += image[j-1+(nx-1)*ny] * Mul;
    numberToadd += image[j+1+(nx-1)*ny] * Mul;
    tmp_image[j+(nx-1)*ny] = numberToadd;
  }

  for (int i = 1; i < nx - 1; i++) {
    for (int j = 1; j < ny - 1; j++) {
      tmp_image[j+i*ny] = image[j+i*ny] * initialMul;
      tmp_image[j+i*ny] += image[j  +(i-1)*ny] * Mul;
      tmp_image[j+i*ny] += image[j  +(i+1)*ny] * Mul;
      tmp_image[j+i*ny] += image[j-1+i*ny] * Mul;
      tmp_image[j+i*ny] += image[j+1+i*ny] * Mul;

      // tmp_image[j+i*ny] = numberToadd;
    }
  }

}

void top_right_corner(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  float initialMul = 0.6f;
  float Mul = 0.1f;
  float numberToadd = 0.0f;

  // when i = nx - 1, j = 0
  numberToadd = image[(nx - 1) * ny] * initialMul;
  numberToadd += image[(ny - 2) * ny] * Mul;
  numberToadd += image[1 + (nx - 1) * ny] * Mul;
  tmp_image[(ny - 1) * ny] = numberToadd;

  // when 0 < i < nx - 1, j = 0
  for (int i = 1; i < nx - 1; i++) {
    numberToadd = image[i * ny] * initialMul;
    numberToadd += image[(i-1) * ny] * Mul;
    numberToadd += image[(i+1) * ny] * Mul;
    numberToadd += image[1 + i * ny] * Mul;
    tmp_image[i * ny] = numberToadd;
  }

  // when i = nx - 1, 0 < j < ny - 1

  for (int j = 1; j < ny - 1; j++) {
    numberToadd = image[j+(nx-1)*ny] * initialMul;
  	numberToadd += image[j+(nx-2)*ny] * Mul;
  	numberToadd += image[j-1+(nx-1)*ny] * Mul;
  	numberToadd += image[j+1+(nx-1)*ny] * Mul;
  	tmp_image[j+(nx-1)*ny] = numberToadd;
  }

  for (int i = 1; i < nx - 1; i++) {
    for (int j = 1; j < ny - 1; j++) {
      tmp_image[j+i*ny] = image[j+i*ny] * initialMul;
      tmp_image[j+i*ny] += image[j  +(i-1)*ny] * Mul;
      tmp_image[j+i*ny] += image[j  +(i+1)*ny] * Mul;
      tmp_image[j+i*ny] += image[j-1+i*ny] * Mul;
      tmp_image[j+i*ny] += image[j+1+i*ny] * Mul;

      // tmp_image[j+i*ny] = numberToadd;
    }
  }

}

void bottom_right_corner(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  float initialMul = 0.6f;
  float Mul = 0.1f;
  float numberToadd = 0.0f;

  // when i = nx - 1,  j = ny - 1
  numberToadd = image[(ny - 1) + (nx - 1) * ny] * initialMul;
  numberToadd += image[(ny - 1) + (nx - 2) * ny] * Mul;
  numberToadd += image[(ny - 2) + (nx - 1) * ny] * Mul;
  tmp_image[(ny - 1) + (nx - 1) * ny] = numberToadd;

  // when i = nx - 1, 0 < j < ny - 1
  for (int j = 1; j < ny - 1; j++) {
    numberToadd = image[j+(nx-1)*ny] * initialMul;
  	numberToadd += image[j+(nx-2)*ny] * Mul;
  	numberToadd += image[j-1+(nx-1)*ny] * Mul;
  	numberToadd += image[j+1+(nx-1)*ny] * Mul;
  	tmp_image[j+(nx-1)*ny] = numberToadd;
  }

  // when 0 < i < nx - 1, j = ny - 1
  for (int i = 1; i < nx - 1; i++) {
    numberToadd = image[(ny - 1) + i * ny] * initialMul;
  	numberToadd += image[(ny - 1) + (i - 1) * ny] * Mul;
  	numberToadd += image[(ny - 1) + (i + 1) * ny] * Mul;
  	numberToadd += image[(ny - 2) + i * ny] * Mul;
  	tmp_image[(ny - 1) + i * ny] = numberToadd;
  }

  for (int i = 1; i < nx - 1; i++) {
    for (int j = 1; j < ny - 1; j++) {
      tmp_image[j+i*ny] = image[j+i*ny] * initialMul;
      tmp_image[j+i*ny] += image[j  +(i-1)*ny] * Mul;
      tmp_image[j+i*ny] += image[j  +(i+1)*ny] * Mul;
      tmp_image[j+i*ny] += image[j-1+i*ny] * Mul;
      tmp_image[j+i*ny] += image[j+1+i*ny] * Mul;

      // tmp_image[j+i*ny] = numberToadd;
    }
  }

}

void stencil(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  float initialMul = 0.6f;

  float  Mul = 0.1f;

  float numberToadd = 0.0f;

  // when i = 0, j = 0
  numberToadd = image[0] * initialMul;
  numberToadd += image[ny] * Mul;
  numberToadd += image[1] * Mul;
  tmp_image[0] = numberToadd;

  // when i = 0, j = ny - 1
  numberToadd = image[(ny - 1)] * initialMul;
  numberToadd += image[(ny - 1) + ny] * Mul;
  numberToadd += image[(ny - 2)] * Mul;
  tmp_image[(ny - 1)] = numberToadd;

  // when i = nx - 1, j = 0
  numberToadd = image[(nx - 1) * ny] * initialMul;
  numberToadd += image[(ny - 2) * ny] * Mul;
  numberToadd += image[1 + (nx - 1) * ny] * Mul;
  tmp_image[(ny - 1) * ny] = numberToadd;

  // when i = nx - 1,  j = ny - 1
  numberToadd = image[(ny - 1) + (nx - 1) * ny] * initialMul;
  numberToadd += image[(ny - 1) + (nx - 2) * ny] * Mul;
  numberToadd += image[(ny - 2) + (nx - 1) * ny] * Mul;
  tmp_image[(ny - 1) + (nx - 1) * ny] = numberToadd;

  // when i = 0, 0 < j < ny - 1, when i = nx - 1, 0 < j < ny - 1

  for (int j = 1; j < ny - 1; j++) {
  	numberToadd = image[j] * initialMul;
  	numberToadd += image[j + ny] * Mul;
  	numberToadd += image[j-1] * Mul;
  	numberToadd += image[j+1] * Mul;
  	tmp_image[j] = numberToadd;
  	numberToadd = image[j+(nx-1)*ny] * initialMul;
  	numberToadd += image[j+(nx-2)*ny] * Mul;
  	numberToadd += image[j-1+(nx-1)*ny] * Mul;
  	numberToadd += image[j+1+(nx-1)*ny] * Mul;
  	tmp_image[j+(nx-1)*ny] = numberToadd;
  }
  // when 0 < i < nx -1, j = 0 and when 0 < i < nx - 1, j = ny - 1

  for (int i = 1; i < nx - 1; i++) {
  	numberToadd = image[i * ny] * initialMul;
  	numberToadd += image[(i-1) * ny] * Mul;
  	numberToadd += image[(i+1) * ny] * Mul;
  	numberToadd += image[1 + i * ny] * Mul;
  	tmp_image[i * ny] = numberToadd;
  	numberToadd = image[(ny - 1) + i * ny] * initialMul;
  	numberToadd += image[(ny - 1) + (i - 1) * ny] * Mul;
  	numberToadd += image[(ny - 1) + (i + 1) * ny] * Mul;
  	numberToadd += image[(ny - 2) + i * ny] * Mul;
  	tmp_image[(ny - 1) + i * ny] = numberToadd;
  }


  for (int i = 1; i < nx - 1; i++) {
    for (int j = 1; j < ny - 1; j++) {
      tmp_image[j+i*ny] = image[j+i*ny] * initialMul;
      tmp_image[j+i*ny] += image[j  +(i-1)*ny] * Mul;
      tmp_image[j+i*ny] += image[j  +(i+1)*ny] * Mul;
      tmp_image[j+i*ny] += image[j-1+i*ny] * Mul;
      tmp_image[j+i*ny] += image[j+1+i*ny] * Mul;

      // tmp_image[j+i*ny] = numberToadd;
    }
  }
}

// Create the input image
void init_image(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  // Zero everything
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      image[j+i*ny] = 0.0;
      tmp_image[j+i*ny] = 0.0;
    }
  }

  // Checkerboard
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      for (int jj = j*ny/8; jj < (j+1)*ny/8; ++jj) {
        for (int ii = i*nx/8; ii < (i+1)*nx/8; ++ii) {
          if ((i+j)%2)
          image[jj+ii*ny] = 100.0;
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char * file_name, const int nx, const int ny, float * restrict image) {

  // Open output file
  FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }


  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  float maximum = 0.0;
  for (int i = 0; i < ny; ++i) {
    for (int j = 0; j < nx; ++j) {
      if (image[j+i*ny] > maximum)
        maximum = image[j+i*ny];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      fputc((char)(255.0*image[j+i*ny]/maximum), fp);
    }
  }

  // Close the file
  fclose(fp);

}

int calc_nrows_from_rank(int rank, int size, int rows) {
  int nrows;

  nrows = rows / 2;

  if ((nrows % 2) != 0) {
    if (rank == size - 1)
      nrows += rows % size;
  }

  return nrows;
}

int calc_ncols_from_rank(int rank, int size, int cols)
{
  int ncols;

  int nsize;

  nsize = size * 0.5;

  ncols = cols / nsize;       /* integer division */
  if (cols % nsize != 0) {  /* if there is a remainder */
    if ((rank == size - 2) || (rank == size - 1)) {
      ncols += cols % nsize;  /* add remainder to last rank */
    }
  }

  return ncols;
}


// Get the current time in seconds since the Epoch
double wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}
