#include <stdio.h>
#include <stdlib.h>
// #include <sys/time.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
  // int ii,jj;             /* row and column indices for the grid */
  // int kk;                /* index for looping over ranks */
  int rank;              /* the rank of this process */
  // int left;              /* the rank of the process to the left */
  // int right;             /* the rank of the process to the right */
  int size;              /* number of processes in the communicator */
  // int tag = 0;           /* scope for adding extra information to a message */
  // MPI_Status status;     /* struct used by MPI_Recv */
  // int local_nrows;       /* number of rows apportioned to this rank */
  // int local_ncols;       /* number of columns apportioned to this rank */
  // int remote_ncols;      /* number of columns apportioned to a remote rank */
  // double *w;             /* local temperature grid at time t     */
  // double *sendbuf;       /* buffer to hold values to send */
  // double *recvbuf;       /* buffer to hold received values */
  // double *printbuf;      /* buffer to hold values for printing */
  char hostname[MPI_MAX_PROCESSOR_NAME];  /* character array to hold hostname running process */
  int strlen;             /* length of a character array */
  int flag;               /* for checking whether MPI_Init() has been called */
  enum bool {FALSE,TRUE}; /* enumerated type: false = 0, true = 1 */

  printf("I'm here\n");


  // initialise our MPI environment
  MPI_Init( &argc, &argv);
  printf("I'm here1\n");

  MPI_Initialized(&flag);
  printf("I'm here2\n");

  if ( flag != TRUE ) {
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }
  printf("I'm here3\n");

  MPI_Get_processor_name(hostname,&strlen);
  printf("I'm here4\n");

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("I'm here5\n");

  MPI_Comm_rank(MPI_COMM_WORLD, &rank );
  printf("I'm here6\n");


  // printf("Hello, world; from host %s: process %d of %d\n", hostname, rank, size);
  printf("Printing\n");
  MPI_Finalize();

  return EXIT_SUCCESS;
}
