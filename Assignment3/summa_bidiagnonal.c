/******************************************************************************************
*
*	Filename:	summa.c
*	Purpose:	A paritally implemented program for MSCS6060 HW. Students will complete 
*			the program by adding SUMMA implementation for matrix multiplication C = A * B.  
*	Assumptions:    A, B, and C are square matrices n by n; 
*			the total number of processors (np) is a square number (q^2).
*	To compile, use 
*	    mpicc -o summa summa.c
*       To run, use
*	    mpiexec -n $(NPROCS) ./summa
*********************************************************************************************/

#include <stdio.h>
#include <time.h>
#include <stdlib.h>	
#include <string.h>
#include <math.h>	
#include "mpi.h"

#define min(a, b) ((a < b) ? a : b)
// #define SZ 4		//Each matrix of entire A, B, and C is SZ by SZ. Set a small value for testing, and set a large value for collecting experimental data.


/**
*   Allocate space for a two-dimensional array
*/
double **alloc_2d_double(int n_rows, int n_cols) {
	int i;
	double **array;
	array = (double **)malloc(n_rows * sizeof (double *));//Double pointer to point to rows start
    array[0] = (double *) malloc(n_rows * n_cols * sizeof(double));//Space allocation for entire matrix
    for (i=1; i<n_rows; i++)
    {
            array[i] = array[0] + i * n_cols;//Defining double pointer start location for 1st index after space alloc.
    }
    return array;
}

/**
*	Initialize arrays A and B with random numbers, and array C with zeros. 
*	Each array is setup as a square block of blck_sz.
**/
void initialize(double **lA, double **lB, double **lC, int blck_sz){
	int i, j;
	double value;
	// double temp_A[6], temp_B[6];
	// Set random values...technically it is already random and this is redundant
	printf("%d\n", blck_sz);
	for(i=0;i<blck_sz;i++)
	{
		
		if(i==0)
		{
			double temp_A[6]={1.0,0.0,0.0,0.0,0.0,0.0};
			double temp_B[6]={1.0,1.0,0.0,0.0,0.0,0.0};

			for (j=0; j<blck_sz; j++)
			{
				lA[i][j] = temp_A[j];
				lB[i][j] = temp_B[j];
			}
		}
		if(i==1)
		{
			double temp_A[6]={1.0,1.0,0.0,0.0,0.0,0.0};
			double temp_B[6]={0.0,1.0,1.0,0.0,0.0,0.0};

			for (j=0; j<blck_sz; j++)
			{
				lA[i][j] = temp_A[j];
				lB[i][j] = temp_B[j];
			}
		}
		if(i==2)
		{
			double temp_A[6]={0.0,1.0,1.0,0.0,0.0,0.0};
			double temp_B[6]={0.0,0.0,1.0,1.0,0.0,0.0};

			for (j=0; j<blck_sz; j++)
			{
				lA[i][j] = temp_A[j];
				lB[i][j] = temp_B[j];
			}
		}
		if(i==3)
		{
			double temp_A[6]={0.0,0.0,1.0,1.0,0.0,0.0};
			double temp_B[6]={0.0,0.0,0.0,1.0,1.0,0.0};

			for (j=0; j<blck_sz; j++)
			{
				lA[i][j] = temp_A[j];
				lB[i][j] = temp_B[j];
			}
		}
		if(i==4)
		{
			double temp_A[6]={0.0,0.0,0.0,1.0,1.0,0.0};
			double temp_B[6]={0.0,0.0,0.0,0.0,1.0,1.0};

			for (j=0; j<blck_sz; j++)
			{
				lA[i][j] = temp_A[j];
				lB[i][j] = temp_B[j];
			}
		}
		if(i==5)
		{
			double temp_A[6]={0.0,0.0,0.0,0.0,1.0,1.0};
			double temp_B[6]={0.0,0.0,0.0,0.0,0.0,1.0};

			for (j=0; j<blck_sz; j++)
			{
				lA[i][j] = temp_A[j];
				lB[i][j] = temp_B[j];
			}
		}
		// memcpy(lA[i], temp_A[0], block_sz*sizeof(double));
		// memcpy(lB[i], temp_B[0], block_sz*sizeof(double));
	}

	printf("\n");
	for (i=0; i<blck_sz; i++)
	{
		for (j=0; j<blck_sz; j++)
		{
			// lA[i][j] = (double)rand() / (double)RAND_MAX;
			
			// lB[i][j] = (double)rand() / (double)RAND_MAX;
			printf("%f ", lB[i][j]);
			lC[i][j] = 0.0;
		}
		printf("\n");
	}
	// printf("\n");
	// for (i=0;i<blck_sz;i++)
	// {
	// 	for (j=0;j<blck_sz;j++)
	// 	{
	// 		printf("%f ", lA[i][j]);
	// 	}
	// 	printf("\n");
	// }

}


/**
*	Perform the SUMMA matrix multiplication. 
*       Follow the pseudo code in lecture slides.
*/
void matmul(double **my_A, double **my_B, double **my_C, int block_sz)//Outer product algo
{	
	for (int k=0;k<block_sz;k++)
	{
		for (int i=0;i<block_sz;i++)
		{
			for (int j=0;j<block_sz;j++)
			{
				// printf("%d\n", my_C[i][j]);
				my_C[i][j]+=(my_A[i][k] * my_B[k][j]);
			}
		}
	}

	// printf("\n");
	// for (int i=0;i<block_sz;i++)
	// {
	// 	for (int j=0;j<block_sz;j++)
	// 	{
	// 		printf("%d ", my_A[i][j]);
	// 	}
	// 	printf("\n");
	// }
}


int main(int argc, char *argv[]) {
	int world_size, rank;						//process rank and total number of processes
	double t1, t2, time_total;	// for timing
	int block_sz;								// Block size length for each processor to handle
	int proc_grid_sz;						// 'q' from the slides
	int SZ=6;

	
	srand(time(NULL));							// Seed random numbers

/* insert MPI functions to 1) start process, 2) get total number of processors and 3) process rank*/
    

    

	
	//Initialize MPI
	MPI_Init(&argc, &argv);
    // Get the number of processes
    
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	/* assign values to 1) proc_grid_sz and 2) block_sz*/

	proc_grid_sz = (int)sqrt((double)world_size);
	block_sz=SZ/proc_grid_sz;
	

	if (SZ % proc_grid_sz != 0)
	{
		printf("Matrix size cannot be evenly split amongst resources!\n");
		printf("Quitting....\n");
		exit(-1);
	}
	// Create the local matrices on each process
	printf("Station0, process: %d", rank);
	double **A, **B, **C;
	A = alloc_2d_double(block_sz, block_sz);
	B = alloc_2d_double(block_sz, block_sz);
	C = alloc_2d_double(block_sz, block_sz);
	int row=0;
	int col=0;
	if(rank!=0)
	{
		row=proc_grid_sz/rank;
		col=proc_grid_sz%rank;
	}
	initialize(A, B, C, block_sz);


    double **buff_A=alloc_2d_double(block_sz, block_sz);
	double **buff_B=alloc_2d_double(block_sz, block_sz);

    //Dimensional Coordinates
	MPI_Comm grid_comm, row_comm, col_comm;
	int dimsizes[2];
	int wraparound[2];
	int coordinates[2];
	int free_coords[2];
	int reorder = 1;
	int my_rank, grid_rank, cart_rank_row, cart_rank_col;
	int row_test, col_test;
	int coordinates_row, coordinates_col;




	dimsizes[0] = dimsizes[1] = proc_grid_sz;
	wraparound[0] = wraparound[1] = 1;
	//Create grid comm
	MPI_Cart_create(MPI_COMM_WORLD, 2, dimsizes, wraparound, reorder, &grid_comm);//int MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[],const int periods[], int reorder, MPI_Comm *comm_cart)

	//Get Rank in grid
	MPI_Comm_rank(grid_comm, &my_rank);

	//Get Co-ordinates in grid
	MPI_Cart_coords(grid_comm, my_rank, 2, coordinates);

	//Get rank in grid coordinates
	MPI_Cart_rank(grid_comm, coordinates, &grid_rank);


	//Free Coordinates for ROW based- multiple columns
	free_coords[0] = 0;//Disable Rows
	free_coords[1] = 1;//Enable Columns
	
	//Split by rules defined above and define row_comm
	MPI_Cart_sub(grid_comm, free_coords, &row_comm);//Create Sub Comm
	MPI_Comm_rank(row_comm, &cart_rank_row); //Rank of Sub Comm
	MPI_Cart_coords(row_comm, cart_rank_row, 1, &coordinates_row);//Coords of Sub Comm

	// for (int i=0;i<2;i++)
	// {
	// 	if(coordinates[1] == i)//If it is my column, then send to all other columns in row
	// 	{
	// 		row_test = my_rank;
	// 		printf("*** %d : element %d\n", my_rank, coordinates_row);
	// 	}
	// 	MPI_Bcast(&row_test, 1, MPI_INT, i, row_comm);//Send to all processes in row
	// 	printf("Process %d > coords = (%d, %d), row_test = %d\n", my_rank, coordinates[0], coordinates[1], row_test);
	// }


	//Free Coordinates for COL based- multiple rows
	free_coords[0] = 1;//Enable Rows
	free_coords[1] = 0;//Disable Columns
	
	//Split by rules defined above and define row_comm
	MPI_Cart_sub(grid_comm, free_coords, &col_comm);//Create Sub Comm
	MPI_Comm_rank(col_comm, &cart_rank_col); //Rank of Sub Comm
	MPI_Cart_coords(col_comm, cart_rank_col, 1, &coordinates_col);//Coords of Sub Comm

	// for (int i=0;i<2;i++)
	// {
	// 	if(coordinates[0] == i)//If it is my row, then send to all other rows in column
	// 	{
	// 		col_test = my_rank;
	// 		printf("*** %d : element %d\n", my_rank, coordinates_col);
	// 	}
	// 	MPI_Bcast(&col_test, 1, MPI_INT, i, col_comm);//Send to all processes in row
	// 	printf("Process %d > coords = (%d, %d), col_test = %d\n", my_rank, coordinates[0], coordinates[1], col_test);
	// }
	// Use MPI_Wtime to get the starting time
	t1 = MPI_Wtime();
	int i, j;
	for ( int k = 0; k < proc_grid_sz; k++)
	{
		

		if(coordinates[1]==k)//Send A to all columns
		{
			// buff_A = (double **)malloc(block_sz * sizeof (double *));
			// memcpy(buff_A[0], A[0], block_sz * sizeof (double *)*block_sz * block_sz * sizeof(double));
			for( i=0;i<block_sz;i++)
			{

				memcpy(buff_A[i], A[i], block_sz*sizeof(double));
			}
			// printf("\n");
			// for (i=0;i<block_sz;i++)
			// {
			// 	for (j=0;j<block_sz;j++)
			// 	{
			// 		printf("%f ", buff_A[i][j]);
			// 	}
			// 	printf("\n");
			// }

		}
		
		MPI_Bcast(&buff_A[0][0], block_sz*block_sz, MPI_DOUBLE, k, row_comm);

		if(coordinates[0]==k)//Send B to all rows
		{
			for( i=0;i<block_sz;i++)
			{

				memcpy(buff_B[i], B[i], block_sz*sizeof(double));
			}
		// 	printf("\n");
		// 	for (i=0;i<block_sz;i++)
		// 	{
		// 		for (j=0;j<block_sz;j++)
		// 		{
		// 			printf("%f ", buff_B[i][j]);
		// 		}
		// 		printf("\n");
		// 	}
		}
		MPI_Bcast(&buff_B[0][0], block_sz*block_sz, MPI_DOUBLE, k, col_comm);

		if(coordinates[0]==k && coordinates[1]==k)
		{
			printf("SAME, process: %d \n", my_rank);

			for (i=0;i<block_sz;i++)
			{
				for (j=0;j<block_sz;j++)
				{
					printf("%f ", C[i][j]);
				}
				printf("\n");
			}
			matmul(A, B, C, block_sz);
			for (i=0;i<block_sz;i++)
			{
				for (j=0;j<block_sz;j++)
				{
					printf("%f ", C[i][j]);
				}
				printf("\n");
			}
		}
		else if(coordinates[0]==k)
		{
			// matmul(buff_A, B, C, block_sz); 
		}
		else if(coordinates[1]==k)
		{
			// matmul(A, buff_B, C, block_sz);
		}
		else
		{
			printf("DIFFERENT, process: %d \n", my_rank);
			
			// for (i=0;i<block_sz;i++)
			// {
			// 	for (j=0;j<block_sz;j++)
			// 	{
			// 		printf("%f ", C[i][j]);
			// 	}
			// 	printf("\n");
			// }
			// matmul(buff_A, buff_B, C, block_sz);
			// for (i=0;i<block_sz;i++)
			// {
			// 	for (j=0;j<block_sz;j++)
			// 	{
			// 		printf("%f ", C[i][j]);
			// 	}
			// 	printf("\n");
			// }
		}
    }




	// Use SUMMA algorithm to calculate product C
	// matmul(rank, proc_grid_sz, block_sz, A, B, C, row, col);

	// Use MPI_Wtime to get the finishing time
	t2 = MPI_Wtime(); 


	// Obtain the elapsed time and assign it to total_time
	time_total=t2-t1;


	
	//Testing



	if (rank == 0)
	{
		// Print in pseudo csv format for easier results compilation
		printf("squareMatrixSideLength,%d,numMPICopies,%d,walltime,%lf\n", SZ, world_size, time_total);
	}

	// Destroy MPI processes

	MPI_Finalize();

	return 0;
}
