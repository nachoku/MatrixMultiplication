/*topfcns.c test basic topology functions

Algorithm:
	1.Build a 2−dimensional Cartesian communicator from MPI_Comm_world
	2.Print topology information for each process
	3.Use MPI_Cart_sub to build a communicator for each row of the Cartesian communicator
	4.Carry out a broadcast across each row communicator
	5.Print results of broadcast
	6.Use MPI_Cart_sub to build a communicator for each column of the Cartesian communicator
	7.Carry out a broadcast across each column  communicator
	8.Print results of broadcast
	
Note:Assumes the number of processes, p, is a perfect square
*/

#include <stdio.h>
#include "mpi.h"
#include <math.h>

int main(int argc, char* argv[])
{

	int p, myrank, q;//p is number of processors

	MPI_Comm grid_comm;
	int dimsizes[2];
	int wraparound[2];
	int coordinates[2];
	int coordinates1;
	int free_coords[2];
	int reorder = 1;
	int my_grid_rank, grid_rank, cart_rank;
	int row_test, col_test;

	MPI_Comm row_comm;
	MPI_Comm col_comm;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	
	q = (int)sqrt((double)p);
	dimsizes[0] = dimsizes[1] = q;
	wraparound[0] = wraparound[1] = 1;
	
	//Create grid comm
	MPI_Cart_create(MPI_COMM_WORLD, 2, dimsizes, wraparound, reorder, &grid_comm);//int MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[],const int periods[], int reorder, MPI_Comm *comm_cart)

	//Get Rank in grid
	MPI_Comm_rank(grid_comm, &my_grid_rank);

	//Get Co-ordinates in grid
	MPI_Cart_coords(grid_comm, my_grid_rank, 2, coordinates);

	//Get rank in grid coordinates
	MPI_Cart_rank(grid_comm, coordinates, &grid_rank);
	
	printf("Process %d > my_grid_rank = %d, coords = (%d, %d), grid_rank = %d\n", myrank, my_grid_rank, coordinates[0], coordinates[1], grid_rank);
	
	//Free Coordinates for ROW based- multiple columns
	free_coords[0] = 0;//Disable Rows
	free_coords[1] = 1;//Enable Columns
	
	//Split by rules defined above and define row_comm
	MPI_Cart_sub(grid_comm, free_coords, &row_comm);//Create Sub Comm
	MPI_Comm_rank(row_comm, &cart_rank); //Rank of Sub Comm
	MPI_Cart_coords(row_comm, cart_rank, 1, &coordinates1);//Coords of Sub Comm

	for (int i=0;i<2;i++)
	{
		if(coordinates[1] == i)//If it is my column, then send to all other columns in row
		{
			row_test = myrank;
			printf("*** %d : element %d\n", myrank, coordinates1);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(&row_test, 1, MPI_INT, i, row_comm);//Send to all processes in row
		MPI_Barrier(MPI_COMM_WORLD);
		printf("Process %d > coords = (%d, %d), row_test = %d\n", myrank, coordinates[0], coordinates[1], row_test);
		MPI_Barrier(MPI_COMM_WORLD);
	}
	
	
	



	//Free Coordinates for COL-multiple Rows
	free_coords[0] = 1;//Enable Rows
	free_coords[1] = 0;//Diable Columns

	//Split by rules defined above and define col comm
	MPI_Cart_sub(grid_comm, free_coords, &col_comm);
	if(coordinates[0] == 0)
		col_test = coordinates[1];
	else
		col_test= -1;
	MPI_Bcast(&col_test, 1, MPI_INT, 0, col_comm);//Send to all processes
	printf("Process %d > coords = (%d, %d), col_test = %d\n", myrank, coordinates[0], coordinates[1], col_test);
	
	MPI_Finalize();
	return 0;
}
