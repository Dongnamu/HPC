
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
double wtime(void);

int main(int argc, char *argv[]) {
  int ii;                /* generic counter */
  int rank;            /* the rank of this process */
  int size;              /* number of processes in the communicator */
  int direction;         /* the coordinate dimension of a shift */
  int disp;              /* displacement, >1 is 'forwards', <1 is 'backwards' along a dimension */
  int dest;
  int source;
  int tag = 0;
  MPI_Status status;

  int north;             /* the rank of the process above this rank in the grid */
  int south;             /* the rank of the process below this rank in the grid */
  int east;              /* the rank of the process to the right of this rank in the grid */
  int west;              /* the rank of the process to the left of this rank in the grid */


  int reorder = 0;       /* an argument to MPI_Cart_create() */
  int dims[N_DIMENSION];       /* array to hold dimensions of an N_DIMENSION grid of processes */
  int periods[N_DIMENSION];    /* array to specificy periodic boundary conditions on each dimension */
  int coords[N_DIMENSION];     /* array to hold the grid coordinates for a rank */
  MPI_Comm comm_cart;    /* a cartesian topology aware communicator */

  int top_right;
  int bottom_right;
  // initialise our MPI environment
  MPI_Init( &argc, &argv);


  // check wheter the initialisation was successful

  // determine the size of the group of processes associated with the 'communicator'.
  // default communicator is MPI_COMM_WORLD, consisting of all the processes in the launched MPI 'job'
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  if (N_DIMENSION != 2) {
    fprintf(stderr, "Error: number of dimension is asuumed to be 2\n");
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

  if (rank == (size - 1)) {
    print("I am here\n");
  }




  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // Allocate the image
  float * restrict image = malloc(sizeof(float)*nx*ny);
  float * restrict tmp_image = malloc(sizeof(float)*nx*ny);

  // Set the input image
  init_image(nx, ny, image, tmp_image);

  // Call the stencil kernel
  double tic = wtime();

  for (int t = 0; t < niters; ++t) {
    stencil(nx, ny, image, tmp_image);
    stencil(nx, ny, tmp_image, image);
  }
  double toc = wtime();


  // Output
  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc-tic);
  printf("------------------------------------\n");

  output_image(OUTPUT_FILE, nx, ny, image);
  free(image);

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

  // when i = nx - 1, j = 0
  numberToadd = image[(nx - 1) * ny] * initialMul;
  numberToadd += image[(ny - 2) * ny] * Mul;
  numberToadd += image[1 + (nx - 1) * ny] * Mul;
  tmp_image[(ny - 1) * ny] = numberToadd;

  // when 0 < i < nx -1, j = 0
  for (int i = 1; i < nx - 1; i++) {
    numberToadd = image[i * ny] * initialMul;
    numberToadd += image[(i-1) * ny] * Mul;
    numberToadd += image[(i+1) * ny] * Mul;
    numberToadd += image[1 + i * ny] * Mul;
    tmp_image[i * ny] = numberToadd;
  }

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

void top(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  float initialMul = 0.6f;
  float Mul = 0.1f;
  float numberToadd = 0.0f;

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

void bottom(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  float initialMul = 0.6f;
  float Mul = 0.1f;
  float numberToadd = 0.0f;

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

  for (int j = 1; j < ny - 1; j++) {
  	numberToadd = image[j] * initialMul;
  	numberToadd += image[j + ny] * Mul;
  	numberToadd += image[j-1] * Mul;
  	numberToadd += image[j+1] * Mul;
  	tmp_image[j] = numberToadd;
  }

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

void bottom_right_corner(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  float initialMul = 0.6f;
  float Mul = 0.1f;
  float numberToadd = 0.0f;

  // when i = nx - 1,  j = ny - 1
  numberToadd = image[(ny - 1) + (nx - 1) * ny] * initialMul;
  numberToadd += image[(ny - 1) + (nx - 2) * ny] * Mul;
  numberToadd += image[(ny - 2) + (nx - 1) * ny] * Mul;
  tmp_image[(ny - 1) + (nx - 1) * ny] = numberToadd;

  for (int j = 1; j < ny - 1; j++) {
    numberToadd = image[j+(nx-1)*ny] * initialMul;
  	numberToadd += image[j+(nx-2)*ny] * Mul;
  	numberToadd += image[j-1+(nx-1)*ny] * Mul;
  	numberToadd += image[j+1+(nx-1)*ny] * Mul;
  	tmp_image[j+(nx-1)*ny] = numberToadd;
  }

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

// Get the current time in seconds since the Epoch
double wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}
