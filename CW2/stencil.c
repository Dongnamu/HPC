
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

void padding(const int rank, const int top_right, const int bottom_right, const int local_ncols, const int local_nrows, const int local_orows, const int local_ocols, float * restrict image, float * restrict image_pad, float* restrict tmp_image_pad);
void process(const int rank, const int north, const int south, const int east, const int west, const int top_right, const int bottom_right, const int niters, const int local_ncols, const int local_nrows, const int local_orows, const int local_ocols, float * restrict image_pad, float * restrict tmp_image_pad, double * sendbuf, double * recvbuf, int tag, MPI_Status status);
void top_left_corner(const int nx, const int ny, float * restrict image, float * restrict tmp_image);
void bottom_left_corner(const int nx, const int ny, float * restrict image, float * restrict tmp_image);
void top(const int nx, const int ny, float * restrict image, float * restrict tmp_image);
void bottom(const int nx, const int ny, float * restrict image, float * restrict tmp_image);
void top_right_corner(const int nx, const int ny, float * restrict image, float * restrict tmp_image);
void bottom_right_corner(const int nx, const int ny, float * restrict image, float * restrict tmp_image);
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
  double *sendbuf;       /* buffer to hold values to send */
  double *recvbuf;       /* buffer to hold received values */
  float* image_pad;
  float* tmp_image_pad;
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

  int top_right;
  int bottom_right;


  int local_nrows;
  int local_ncols;
  int local_usual_ncols;
  int local_usual_nrows;
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

  // MPI_Dims_create(size, N_DIMENSION, dims);
  // if(rank == MASTER) {
  //   printf("ranks spread over a grid of %d dimension(s): [%d,%d]\n", N_DIMENSION, dims[0], dims[1]);
  // }

  dims[0] = size / 2;
  dims[1] = 2;
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
  // printf("rank: %d\n\tnorth=%d\n\tsouth=%d\n\teast=%d\n\twest=%d\n", rank,north,south,east,west);

  top_right = size - 2;
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

  local_nrows = calc_nrows_from_rank(rank, size, ny);
  local_ncols = calc_ncols_from_rank(rank, size, nx);

  int local_orows = local_nrows + 1;
  int local_ocols = local_ncols + 1;

  float * restrict image_original = malloc(sizeof(float) * nx * ny);
  float * restrict tmp_image_original = malloc(sizeof(float) * nx * ny);


  init_image(nx, ny, image_original, tmp_image_original);

  local_usual_ncols = nx / (size * 0.5);
  local_usual_nrows = ny / 2;

  if ((rank % 2) != 1) {
    loop_row_start_point = 0;
    loop_row_end_point = local_usual_nrows;
  } else {
    loop_row_start_point = local_usual_nrows;
    loop_row_end_point = ny;
  }

  loop_col_start_point = (rank / 2) * local_usual_ncols;
  loop_col_end_point = loop_col_start_point + local_ncols;

  // memory allocation
  float * restrict image = malloc(sizeof(float) * local_nrows * local_ncols);
  float * restrict tmp_image = malloc(sizeof(float) * local_nrows * local_ncols);

  if (rank == MASTER || rank == 1 || rank == top_right || rank == bottom_right) {
    image_pad = (float*) malloc(sizeof(float) * (local_orows) * (local_ocols));
    tmp_image_pad = (float*) malloc(sizeof(float) * (local_orows) * (local_ocols));

    for (int i = 0; i < local_orows; i++) {
      for (int j = 0; j < local_ocols; j++) {
        image_pad[j + i * local_ocols] = 0;
        tmp_image_pad[j + i * local_ocols] = 0;
      }
    }
  } else {
    image_pad = (float*) malloc(sizeof(float) * (local_orows) * (local_ncols + 2));
    tmp_image_pad = (float*) malloc(sizeof(float) * (local_orows) * (local_ncols + 2));
    for (int i = 0; i < local_orows; i++) {
      for (int j = 0; j < local_ncols + 2; j++) {
        image_pad[j + i * (local_ncols + 2)] = 0;
        tmp_image_pad[j + i * (local_ncols + 2)] = 0;
      }
    }
  }

  // Splitting image
  for (int i = loop_row_start_point; i < loop_row_end_point;  i++) {
    for (int j = loop_col_start_point; j < loop_col_end_point; j++) {
      image[(j - loop_col_start_point) + (i - loop_row_start_point) * local_ncols] = image_original[j + i * nx];
      tmp_image[(j - loop_col_start_point) + (i - loop_row_start_point) * local_ncols] = tmp_image_original[j + i * nx];
    }
  }


  sendbuf = (double*)malloc(sizeof(double) * (local_orows));
  recvbuf = (double*)malloc(sizeof(double) * (local_orows));

  //padding
  padding(rank, top_right, bottom_right, local_ncols, local_nrows, local_orows, local_ocols, image, image_pad, tmp_image_pad);

  //start timer
  double tic = wtime();
  //communication & process
  process(rank, north, south, east, west, top_right, bottom_right, niters, local_ncols, local_nrows, local_orows, local_ocols, image_pad, tmp_image_pad, sendbuf, recvbuf, tag, status);

  double toc = wtime();

  if (rank == MASTER) {
    for (int i = 0; i < local_nrows; i++) {
      for (int j = 0; j < local_ncols; j++) {
        tmp_image_original[j + i * nx] = image_pad[j + i * local_ocols];
      }
    }

    int rank_rows;
    int rank_cols;
    int rank_start_row;
    int rank_start_col;

    for (int k = 1; k < size; k++) {
      MPI_Recv(recvbuf, 4, MPI_DOUBLE, k, tag, MPI_COMM_WORLD, &status);

      rank_rows = recvbuf[0];
      rank_cols = recvbuf[1];
      rank_start_row = recvbuf[2];
      rank_start_col = recvbuf[3];

      for (int i = 0; i < rank_rows; i++) {
        MPI_Recv(recvbuf, rank_rows, MPI_DOUBLE, k, tag, MPI_COMM_WORLD, &status);
        for (int j = 0; j < rank_cols; j++) {
          tmp_image_original[(j + rank_start_col ) + (i + rank_start_row) * nx] = recvbuf[j];
        }
      }
    }
    output_image("output.pgm", nx, ny, tmp_image_original);
  } else {
    if (rank == 1) {
      sendbuf[0] = local_nrows;
      sendbuf[1] = local_ncols;
      sendbuf[2] = loop_row_start_point;
      sendbuf[3] = loop_col_start_point;
      MPI_Send(sendbuf, 4, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD);

      for (int i = 1; i < local_orows; i++) {
        for (int j = 0; j < local_ncols; j++) {
          sendbuf[j] = image_pad[j + i * local_ocols];
        }
        MPI_Send(sendbuf, local_nrows, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD);
      }
    } else {
      if (rank == top_right) {
        sendbuf[0] = local_nrows;
        sendbuf[1] = local_ncols;
        sendbuf[2] = loop_row_start_point;
        sendbuf[3] = loop_col_start_point;
        MPI_Send(sendbuf, 4, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD);

        for (int i = 0; i < local_nrows; i++) {
          for (int j = 1; j < local_ocols; j++) {
            sendbuf[(j - 1)] = image_pad[j + i * local_ocols];
          }
          MPI_Send(sendbuf, local_nrows, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD);
        }

      } else {
        if (rank == bottom_right) {
          sendbuf[0] = local_nrows;
          sendbuf[1] = local_ncols;
          sendbuf[2] = loop_row_start_point;
          sendbuf[3] = loop_col_start_point;
          MPI_Send(sendbuf, 4, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD);

          for (int i = 1; i < local_orows; i++) {
            for (int j = 1; j < local_ocols; j++) {
              sendbuf[(j - 1)] = image_pad[j + i * local_ocols];
            }
            MPI_Send(sendbuf, local_nrows, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD);
          }
        } else {
          if ((rank % 2) == 0) {
            sendbuf[0] = local_nrows;
            sendbuf[1] = local_ncols;
            sendbuf[2] = loop_row_start_point;
            sendbuf[3] = loop_col_start_point;
            MPI_Send(sendbuf, 4, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD);

            for (int i = 0; i < local_nrows; i++) {
              for (int j = 1; j < local_ocols; j++) {
                sendbuf[(j - 1)] = image_pad[j + i * (local_ncols + 2)];
              }
              MPI_Send(sendbuf, local_nrows, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD);
            }
          } else {
            if ((rank % 2) == 1) {
              sendbuf[0] = local_nrows;
              sendbuf[1] = local_ncols;
              sendbuf[2] = loop_row_start_point;
              sendbuf[3] = loop_col_start_point;
              MPI_Send(sendbuf, 4, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD);

              for (int i = 1; i < local_orows; i++) {
                for (int j = 1; j < local_ocols; j++) {
                  sendbuf[(j - 1)] = image_pad[j + i * (local_ncols + 2)];
                }
                MPI_Send(sendbuf, local_nrows, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD);
              }
            }
          }
        }
      }
    }
  }

  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc-tic);
  printf("------------------------------------\n");


  // if (rank == 2){
    // output_image("RANK2.pgm", local_nrows, local_ncols, image_pad);
  // }

  free(image);
  free(tmp_image);
  free(image_original);
  free(tmp_image_original);
  free(image_pad);
  free(tmp_image_pad);
  free(sendbuf);
  free(recvbuf);

  MPI_Finalize();

  return EXIT_SUCCESS;
}

void padding(const int rank, const int top_right, const int bottom_right, const int local_ncols, const int local_nrows, const int local_orows, const int local_ocols, float * restrict image, float * restrict image_pad, float* restrict tmp_image_pad) {
  if (rank == MASTER) {
    for (int i = 0; i < local_nrows; i++) {
      for (int j = 0; j < local_ncols; j++) {
        image_pad[j + i * (local_ocols)] = image[j + i * local_ncols];
      }
    }
  } else {
    if (rank == 1) {
      for (int i = 1; i < local_orows; i++) {
        for (int j = 0; j < local_ncols; j++) {
          image_pad[j + i * (local_ocols)] = image[j + (i - 1) * local_ncols];
        }
      }
    } else {
      if (rank == top_right) {
        for (int i = 0; i < local_nrows; i++) {
          for (int j = 1; j < local_ocols; j++) {
            image_pad[j + i * (local_ocols)] = image[(j - 1) + i * local_ncols];
          }
        }
      } else {
        if (rank == bottom_right) {
          for (int i = 1; i < local_orows; i++) {
            for (int j = 1; j < local_ocols; j++) {
              image_pad[j + i * (local_ocols)] = image[(j - 1) + (i - 1) * local_ncols];
            }
          }
        } else {
          if ((rank % 2) == 0) {
            for (int i = 0; i < local_nrows; i++) {
              for (int j = 1; j < local_ocols; j++) {
                image_pad[j + i * (local_ncols + 2)] = image[(j - 1) + i * local_ncols];
              }
            }
          } else {
            if ((rank % 2) == 1) {
              for (int i = 1; i < local_orows; i++) {
                for (int j = 1; j < local_ocols; j++) {
                  image_pad[j + i * (local_ncols + 2)] = image[(j - 1) + (i - 1) * local_ncols];
                }
              }
            }
          }
        }
      }
    }
  }
}

void process(const int rank, const int north, const int south, const int east, const int west, const int top_right, const int bottom_right, const int niters, const int local_ncols, const int local_nrows, const int local_orows, const int local_ocols, float * restrict image_pad, float * restrict tmp_image_pad, double * sendbuf, double * recvbuf, int tag, MPI_Status status) {
  for (int t = 0; t < niters; ++t) {
    if (rank == MASTER) {

      //first
      for (int j = 0; j < local_ncols; j++) {
        sendbuf[j] = image_pad[j + (local_nrows - 1) * (local_ocols)];
      }

      MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, south, tag, recvbuf, (local_orows), MPI_DOUBLE, south, tag, MPI_COMM_WORLD, &status);

      for (int j = 0; j < local_ncols; j++) {
        image_pad[j + (local_nrows) * (local_ocols)] = recvbuf[j];
      }

      for (int i = 0; i < local_nrows; i++) {
        sendbuf[i] = image_pad[(local_ncols - 1) + i * (local_ocols)];
      }

      MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, east, tag, recvbuf, (local_orows), MPI_DOUBLE, east, tag, MPI_COMM_WORLD, &status);

      for (int i = 0; i < local_nrows; i++) {
          image_pad[local_ncols + i * (local_ocols)] = recvbuf[i];
      }

      top_left_corner((local_ocols), (local_orows), image_pad, tmp_image_pad);

      //second
      for (int j = 0; j < local_ncols; j++) {
        sendbuf[j] = tmp_image_pad[j + (local_nrows - 1) * (local_ocols)];
      }

      MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, south, tag, recvbuf, (local_orows), MPI_DOUBLE, south, tag, MPI_COMM_WORLD, &status);

      for (int j = 0; j < local_ncols; j++) {
        tmp_image_pad[j + (local_nrows) * (local_ocols)] = recvbuf[j];
      }

      for (int i = 0; i < local_nrows; i++) {
        sendbuf[i] = tmp_image_pad[(local_ncols - 1) + i * (local_ocols)];
      }

      MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, east, tag, recvbuf, (local_orows), MPI_DOUBLE, east, tag, MPI_COMM_WORLD, &status);

      for (int i = 0; i < local_nrows; i++) {
          tmp_image_pad[local_ncols + i * (local_ocols)] = recvbuf[i];
      }
      top_left_corner((local_ocols), (local_orows), tmp_image_pad, image_pad);

      // for (int i = 0; i < local_nrows; i++) {
      //   for (int j = 0; j < local_ncols; j++) {
      //     printf(" %f |", image_pad[j + i * local_ocols]);
      //   }
      //   printf("\n");
      // }
      // printf("\n");

      // output_image("RANK0.pgm", local_ocols, local_orows, image_pad);

    } else {
      if (rank == 1) {

        //first
        for (int j = 0; j < local_ncols; j++) {
          sendbuf[j] = image_pad[j + local_ocols];
        }

        MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, north, tag, recvbuf, (local_orows), MPI_DOUBLE, north, tag, MPI_COMM_WORLD, &status);

        for (int j = 0; j < local_ncols; j++) {
          image_pad[j] = recvbuf[j];

        }

        for (int i = 1; i < local_orows; i++) {
          sendbuf[i - 1] = image_pad[(local_ncols - 1) + i * (local_ocols)];
        }

        MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, east, tag, recvbuf, (local_orows), MPI_DOUBLE, east, tag, MPI_COMM_WORLD, &status);

        for (int i = 1; i < local_orows; i++) {
          image_pad[local_ncols + i * (local_ocols)] = recvbuf[i - 1];
        }

        bottom_left_corner((local_ocols), (local_orows), image_pad, tmp_image_pad);

        //seconds
        for (int j = 0; j < local_ncols; j++) {
          sendbuf[j] = tmp_image_pad[j + local_ocols];
        }

        MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, north, tag, recvbuf, (local_orows), MPI_DOUBLE, north, tag, MPI_COMM_WORLD, &status);

        for (int j = 0; j < local_ncols; j++) {
          tmp_image_pad[j] = recvbuf[j];
        }

        for (int i = 1; i < local_orows; i++) {
          sendbuf[i - 1] = tmp_image_pad[(local_ncols - 1) + i * (local_ocols)];
        }

        MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, east, tag, recvbuf, (local_orows), MPI_DOUBLE, east, tag, MPI_COMM_WORLD, &status);

        for (int i = 1; i < local_orows; i++) {
          tmp_image_pad[local_ncols + i * (local_ocols)] = recvbuf[i - 1];
        }
        bottom_left_corner((local_ocols), (local_orows), tmp_image_pad, image_pad);


        // output_image("RANK1.pgm", local_ocols, local_orows, image_pad);

      } else {
        if (rank == top_right) {
          //first
          for (int i = 0; i < local_nrows; i++) {
            sendbuf[i] = image_pad[1 + i * (local_ocols)];
          }

          MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, west, tag, recvbuf, (local_orows), MPI_DOUBLE, west, tag, MPI_COMM_WORLD, &status);

          for (int i = 0; i < local_nrows; i++) {
            image_pad[i * (local_ocols)] = recvbuf[i];
          }

          for (int j = 1; j < local_ocols; j++) {
            sendbuf[j - 1] = image_pad[j + (local_nrows - 1) * (local_ocols)];
          }

          MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, south, tag, recvbuf, (local_orows), MPI_DOUBLE, south, tag, MPI_COMM_WORLD, &status);

          for (int j = 1; j < (local_ocols); j++) {
            image_pad[j + local_nrows * (local_ocols)] = recvbuf[j - 1];
          }

          top_right_corner((local_ocols), (local_orows), image_pad, tmp_image_pad);

          //seconds
          for (int i = 0; i < local_nrows; i++) {
            sendbuf[i] = tmp_image_pad[1 + i * (local_ocols)];
          }

          MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, west, tag, recvbuf, (local_orows), MPI_DOUBLE, west, tag, MPI_COMM_WORLD, &status);

          for (int i = 0; i < local_nrows; i++) {
            tmp_image_pad[i * (local_ocols)] = recvbuf[i];
          }

          for (int j = 1; j < local_ocols; j++) {
            sendbuf[j - 1] = tmp_image_pad[j + (local_nrows - 1) * (local_ocols)];
          }

          MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, south, tag, recvbuf, (local_orows), MPI_DOUBLE, south, tag, MPI_COMM_WORLD, &status);

          for (int j = 1; j < (local_ocols); j++) {
            tmp_image_pad[j + local_nrows * (local_ocols)] = recvbuf[j - 1];
          }
          top_right_corner((local_ocols), (local_orows), tmp_image_pad, image_pad);

          // output_image("RANK4.pgm", local_ocols, local_orows, image_pad);

        } else {
          if (rank == bottom_right) {
            //first
            for (int j = 1; j < local_ocols; j++) {
              sendbuf[j - 1] = image_pad[j + local_ocols];
            }

            MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, north, tag, recvbuf, (local_orows), MPI_DOUBLE, north, tag, MPI_COMM_WORLD, &status);

            for (int j = 1; j < local_ocols; j++) {
              image_pad[j] = recvbuf[j - 1];

            }

            for (int i = 1; i < local_orows; i++) {
              sendbuf[i - 1] = image_pad[1 + i * (local_ocols)];
            }

            MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, west, tag, recvbuf, (local_orows), MPI_DOUBLE, west, tag, MPI_COMM_WORLD, &status);

            for (int i = 1; i < local_orows; i++) {
              image_pad[i * (local_ocols)] = recvbuf[i - 1];

            }

            bottom_right_corner((local_ocols), (local_orows), image_pad, tmp_image_pad);

            //second
            for (int j = 1; j < local_ocols; j++) {
              sendbuf[j - 1] = tmp_image_pad[j + local_ocols];
            }

            MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, north, tag, recvbuf, (local_orows), MPI_DOUBLE, north, tag, MPI_COMM_WORLD, &status);

            for (int j = 1; j < local_ocols; j++) {
              tmp_image_pad[j] = recvbuf[j - 1];

            }

            for (int i = 1; i < local_orows; i++) {
              sendbuf[i - 1] = tmp_image_pad[1 + i * (local_ocols)];
            }

            MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, west, tag, recvbuf, (local_orows), MPI_DOUBLE, west, tag, MPI_COMM_WORLD, &status);

            for (int i = 1; i < local_orows; i++) {
              tmp_image_pad[i * (local_ocols)] = recvbuf[i - 1];

            }
            bottom_right_corner((local_ocols), (local_orows), tmp_image_pad, image_pad);

            // output_image("RANK5.pgm", local_ocols, local_orows, image_pad);

          } else {
            if ((rank % 2) == 0) {

              //first
              //send to south
              for (int j = 1; j < local_ocols; j++) {
                sendbuf[j - 1] = image_pad[j + (local_nrows - 1) * (local_ncols + 2)];
              }

              MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, south, tag, recvbuf, (local_orows), MPI_DOUBLE, south, tag, MPI_COMM_WORLD, &status);

              for (int j = 1; j < local_ocols; j++) {
                image_pad[j + (local_nrows) * (local_ncols + 2)] = recvbuf[j - 1];

              }

              // send to west
              for (int i = 0; i < local_nrows; i++) {
                sendbuf[i] = image_pad[1 + i * (local_ncols + 2)];
              }

              MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, west, tag, recvbuf, (local_orows), MPI_DOUBLE, west, tag, MPI_COMM_WORLD, &status);

              for (int i = 0; i < local_nrows; i++) {
                image_pad[i * (local_ncols + 2)] = recvbuf[i];

              }

              // send to east

              for (int i = 0; i < local_nrows; i++) {
                sendbuf[i] = image_pad[(local_ncols) + i * (local_ncols + 2)];
              }

              MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, east, tag, recvbuf, (local_orows), MPI_DOUBLE, east, tag, MPI_COMM_WORLD, &status);

              for (int i = 0; i < local_nrows; i++) {
                  image_pad[(local_ocols) + i * (local_ncols + 2)] = recvbuf[i];

              }

              top((local_ncols + 2), (local_orows), image_pad, tmp_image_pad);

              //seconds
              //send to south
              for (int j = 1; j < local_ocols; j++) {
                sendbuf[j - 1] = tmp_image_pad[j + (local_nrows - 1) * (local_ncols + 2)];
              }

              MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, south, tag, recvbuf, (local_orows), MPI_DOUBLE, south, tag, MPI_COMM_WORLD, &status);

              for (int j = 1; j < local_ocols; j++) {
                tmp_image_pad[j + (local_nrows) * (local_ncols + 2)] = recvbuf[j - 1];

              }

              // send to west
              for (int i = 0; i < local_nrows; i++) {
                sendbuf[i] = tmp_image_pad[1 + i * (local_ncols + 2)];
              }

              MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, west, tag, recvbuf, (local_orows), MPI_DOUBLE, west, tag, MPI_COMM_WORLD, &status);

              for (int i = 0; i < local_nrows; i++) {
                tmp_image_pad[i * (local_ncols + 2)] = recvbuf[i];

              }

              // send to east

              for (int i = 0; i < local_nrows; i++) {
                sendbuf[i] = tmp_image_pad[(local_ncols) + i * (local_ncols + 2)];
              }

              MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, east, tag, recvbuf, (local_orows), MPI_DOUBLE, east, tag, MPI_COMM_WORLD, &status);

              for (int i = 0; i < local_nrows; i++) {
                  tmp_image_pad[(local_ocols) + i * (local_ncols + 2)] = recvbuf[i];

              }
              top((local_ncols + 2), (local_orows), tmp_image_pad, image_pad);

              // output_image("RANK2.pgm", local_ncols + 2, local_orows, image_pad);

            } else {
              if ((rank % 2) == 1) {
                //first
                //send to north
                for (int j = 1; j < local_ocols; j++) {
                  sendbuf[j - 1] = image_pad[j + local_ncols + 2];
                }

                MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, north, tag, recvbuf, (local_orows), MPI_DOUBLE, north, tag, MPI_COMM_WORLD, &status);

                for (int j = 1; j < local_ocols; j++) {
                  image_pad[j] = recvbuf[j - 1];

                }

                // send to west
                for (int i = 1; i < local_orows; i++) {
                  sendbuf[i - 1] = image_pad[1 + i * (local_ncols + 2)];
                }

                MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, west, tag, recvbuf, (local_orows), MPI_DOUBLE, west, tag, MPI_COMM_WORLD, &status);

                for (int i = 1; i < local_orows; i++) {
                  image_pad[i * (local_ncols + 2)] = recvbuf[i - 1];

                }

                // send to east

                for (int i = 1; i < local_orows; i++) {
                  sendbuf[i - 1] = image_pad[(local_ncols) + i * (local_ncols + 2)];
                }

                MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, east, tag, recvbuf, (local_orows), MPI_DOUBLE, east, tag, MPI_COMM_WORLD, &status);

                for (int i = 1; i < local_orows; i++) {
                    image_pad[(local_ocols) + i * (local_ncols + 2)] = recvbuf[i - 1];

                }

                bottom((local_ncols + 2), (local_orows), image_pad, tmp_image_pad);

                //second
                //send to north
                for (int j = 1; j < local_ocols; j++) {
                  sendbuf[j - 1] = tmp_image_pad[j + local_ncols + 2];
                }

                MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, north, tag, recvbuf, (local_orows), MPI_DOUBLE, north, tag, MPI_COMM_WORLD, &status);

                for (int j = 1; j < local_ocols; j++) {
                  tmp_image_pad[j] = recvbuf[j - 1];

                }

                // send to west
                for (int i = 1; i < local_orows; i++) {
                  sendbuf[i - 1] = tmp_image_pad[1 + i * (local_ncols + 2)];
                }

                MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, west, tag, recvbuf, (local_orows), MPI_DOUBLE, west, tag, MPI_COMM_WORLD, &status);

                for (int i = 1; i < local_orows; i++) {
                  tmp_image_pad[i * (local_ncols + 2)] = recvbuf[i - 1];

                }

                // send to east

                for (int i = 1; i < local_orows; i++) {
                  sendbuf[i - 1] = tmp_image_pad[(local_ncols) + i * (local_ncols + 2)];
                }

                MPI_Sendrecv(sendbuf, (local_orows), MPI_DOUBLE, east, tag, recvbuf, (local_orows), MPI_DOUBLE, east, tag, MPI_COMM_WORLD, &status);

                for (int i = 1; i < local_orows; i++) {
                    tmp_image_pad[(local_ocols) + i * (local_ncols + 2)] = recvbuf[i - 1];

                }
                bottom((local_ncols + 2), (local_orows), tmp_image_pad, image_pad);

                // output_image("RANK3.pgm", local_ncols + 2, local_orows, image_pad);

              }
            }
          }
        }
      }
    }
  }
}


void top_left_corner(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  float initialMul = 0.6f;
  float Mul = 0.1f;
  float numberToadd = 0.0f;

  // when i = 0, j = 0
  numberToadd = image[0] * initialMul;
  numberToadd += image[nx] * Mul;
  numberToadd += image[1] * Mul;
  tmp_image[0] = numberToadd;

  for (int j = 1; j < nx - 1; j++) {
    numberToadd = image[j] * initialMul;
  	numberToadd += image[j + nx] * Mul;
  	numberToadd += image[j-1] * Mul;
  	numberToadd += image[j+1] * Mul;
  	tmp_image[j] = numberToadd;
  }

  for (int i = 1; i < ny - 1; i++) {
    numberToadd = image[i * nx] * initialMul;
  	numberToadd += image[(i-1) * nx] * Mul;
  	numberToadd += image[(i+1) * nx] * Mul;
  	numberToadd += image[1 + i * nx] * Mul;
  	tmp_image[i * nx] = numberToadd;
  }

  for (int i = 1; i < ny - 1; i++) {
    for (int j = 1; j < nx - 1; j++) {
      tmp_image[j+i*nx] = image[j+i*nx] * initialMul;
      tmp_image[j+i*nx] += image[j  +(i-1)*nx] * Mul;
      tmp_image[j+i*nx] += image[j  +(i+1)*nx] * Mul;
      tmp_image[j+i*nx] += image[j-1+i*nx] * Mul;
      tmp_image[j+i*nx] += image[j+1+i*nx] * Mul;
    }
  }
}

void bottom_left_corner(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  float initialMul = 0.6f;
  float Mul = 0.1f;
  float numberToadd = 0.0f;

  numberToadd = image[(ny - 1) * nx] * initialMul;
  numberToadd += image[(ny - 2) * nx] * Mul;
  numberToadd += image[1 + (ny - 1) * nx] * Mul;
  tmp_image[(ny - 1) * nx] = numberToadd;

  for (int i = 1; i < ny - 1; i++) {
  	numberToadd = image[i * nx] * initialMul;
  	numberToadd += image[(i-1) * nx] * Mul;
  	numberToadd += image[(i+1) * nx] * Mul;
  	numberToadd += image[1 + i * nx] * Mul;
  	tmp_image[i * nx] = numberToadd;
  }

  for (int j = 1; j < nx - 1; j++) {
  	numberToadd = image[j+(ny-1)*nx] * initialMul;
  	numberToadd += image[j+(ny-2)*nx] * Mul;
  	numberToadd += image[j-1+(ny-1)*nx] * Mul;
  	numberToadd += image[j+1+(ny-1)*nx] * Mul;
  	tmp_image[j+(ny-1)*nx] = numberToadd;
  }

  for (int i = 1; i < ny - 1; i++) {
    for (int j = 1; j < nx - 1; j++) {
      tmp_image[j+i*nx] = image[j+i*nx] * initialMul;
      tmp_image[j+i*nx] += image[j  +(i-1)*nx] * Mul;
      tmp_image[j+i*nx] += image[j  +(i+1)*nx] * Mul;
      tmp_image[j+i*nx] += image[j-1+i*nx] * Mul;
      tmp_image[j+i*nx] += image[j+1+i*nx] * Mul;
    }
  }

}

void top(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  float initialMul = 0.6f;
  float Mul = 0.1f;
  float numberToadd = 0.0f;

  for (int j = 1; j < nx - 1; j++) {
    numberToadd = image[j] * initialMul;
    numberToadd += image[j + nx] * Mul;
    numberToadd += image[j-1] * Mul;
    numberToadd += image[j+1] * Mul;
    tmp_image[j] = numberToadd;
  }

  for (int i = 1; i < ny - 1; i++) {
    for (int j = 1; j < nx - 1; j++) {
      tmp_image[j+i*nx] = image[j+i*nx] * initialMul;
      tmp_image[j+i*nx] += image[j  +(i-1)*nx] * Mul;
      tmp_image[j+i*nx] += image[j  +(i+1)*nx] * Mul;
      tmp_image[j+i*nx] += image[j-1+i*nx] * Mul;
      tmp_image[j+i*nx] += image[j+1+i*nx] * Mul;

      // tmp_image[j+i*ny] = numberToadd;
    }
  }

}

void bottom(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  float initialMul = 0.6f;
  float Mul = 0.1f;
  float numberToadd = 0.0f;

  for (int j = 1; j < nx - 1; j++) {
    numberToadd = image[j+(ny-1)*nx] * initialMul;
  	numberToadd += image[j+(ny-2)*nx] * Mul;
  	numberToadd += image[j-1+(ny-1)*nx] * Mul;
  	numberToadd += image[j+1+(ny-1)*nx] * Mul;
  	tmp_image[j+(ny-1)*nx] = numberToadd;
  }


  for (int i = 1; i < ny - 1; i++) {
    for (int j = 1; j < nx - 1; j++) {
      tmp_image[j+i*nx] = image[j+i*nx] * initialMul;
      tmp_image[j+i*nx] += image[j  +(i-1)*nx] * Mul;
      tmp_image[j+i*nx] += image[j  +(i+1)*nx] * Mul;
      tmp_image[j+i*nx] += image[j-1+i*nx] * Mul;
      tmp_image[j+i*nx] += image[j+1+i*nx] * Mul;

      // tmp_image[j+i*ny] = numberToadd;
    }
  }

}

void top_right_corner(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  float initialMul = 0.6f;
  float Mul = 0.1f;
  float numberToadd = 0.0f;

  numberToadd = image[(nx - 1)] * initialMul;
  numberToadd += image[(nx - 1) + nx] * Mul;
  numberToadd += image[(nx - 2)] * Mul;
  tmp_image[(nx - 1)] = numberToadd;

  for (int j = 1; j < nx - 1; j++) {
  	numberToadd = image[j] * initialMul;
  	numberToadd += image[j + nx] * Mul;
  	numberToadd += image[j-1] * Mul;
  	numberToadd += image[j+1] * Mul;
  	tmp_image[j] = numberToadd;
  }

  for (int i = 1; i < ny - 1; i++) {
    numberToadd = image[(nx - 1) + i * nx] * initialMul;
  	numberToadd += image[(nx - 1) + (i - 1) * nx] * Mul;
  	numberToadd += image[(nx - 1) + (i + 1) * nx] * Mul;
  	numberToadd += image[(nx - 2) + i * nx] * Mul;
  	tmp_image[(nx - 1) + i * nx] = numberToadd;
  }

  for (int i = 1; i < ny - 1; i++) {
    for (int j = 1; j < nx - 1; j++) {
      tmp_image[j+i*nx] = image[j+i*nx] * initialMul;
      tmp_image[j+i*nx] += image[j  +(i-1)*nx] * Mul;
      tmp_image[j+i*nx] += image[j  +(i+1)*nx] * Mul;
      tmp_image[j+i*nx] += image[j-1+i*nx] * Mul;
      tmp_image[j+i*nx] += image[j+1+i*nx] * Mul;
    }
  }


}

void bottom_right_corner(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  float initialMul = 0.6f;
  float Mul = 0.1f;
  float numberToadd = 0.0f;

  numberToadd = image[(nx - 1) + (ny - 1) * nx] * initialMul;
  numberToadd += image[(nx - 1) + (ny - 2) * nx] * Mul;
  numberToadd += image[(nx - 2) + (ny - 1) * nx] * Mul;
  tmp_image[(nx - 1) + (ny - 1) * nx] = numberToadd;

  for (int i = 1; i < ny - 1; i++) {
    numberToadd = image[(nx - 1) + i * nx] * initialMul;
  	numberToadd += image[(nx - 1) + (i - 1) * nx] * Mul;
  	numberToadd += image[(nx - 1) + (i + 1) * nx] * Mul;
  	numberToadd += image[(nx - 2) + i * nx] * Mul;
  	tmp_image[(nx - 1) + i * nx] = numberToadd;
  }

  for (int j = 1; j < nx - 1; j++) {
    numberToadd = image[j+(ny-1)*nx] * initialMul;
  	numberToadd += image[j+(ny-2)*nx] * Mul;
  	numberToadd += image[j-1+(ny-1)*nx] * Mul;
  	numberToadd += image[j+1+(ny-1)*nx] * Mul;
  	tmp_image[j+(ny-1)*nx] = numberToadd;
  }

  for (int i = 1; i < ny - 1; i++) {
    for (int j = 1; j < nx - 1; j++) {
      tmp_image[j+i*nx] = image[j+i*nx] * initialMul;
      tmp_image[j+i*nx] += image[j  +(i-1)*nx] * Mul;
      tmp_image[j+i*nx] += image[j  +(i+1)*nx] * Mul;
      tmp_image[j+i*nx] += image[j-1+i*nx] * Mul;
      tmp_image[j+i*nx] += image[j+1+i*nx] * Mul;
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
      if (image[j+i*nx] > maximum)
        maximum = image[j+i*ny];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      fputc((char)(255.0*image[i+j*nx]/maximum), fp);
    }
  }

  // Close the file
  fclose(fp);

}

int calc_nrows_from_rank(int rank, int size, int rows) {
  int nrows;

  int nsize = 2;

  nrows = rows / nsize;

  if ((rows % nsize) != 0) {
    if (rank == (size - 2) || rank == (size - 1))
      nrows += rows % nsize;
  }

  return nrows;
}

int calc_ncols_from_rank(int rank, int size, int cols)
{
  int ncols;

  int nsize = size * 0.5;

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
