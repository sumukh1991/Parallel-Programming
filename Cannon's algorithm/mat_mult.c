/****
 * Author : Hallymysore Ravindra, Sumukh
 * Date	  : 04/13/2016
 * Desc   : Matrix multiplication using distributed memory - MPI library
 ****/


#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"
#include "time.h"
#include "math.h"

#define MATRIX_SIZE 1024

// Matrix row rotation : Cannon's algorithm
void matrix_row_rotate(int rank, int gridSize, int* matAlpha, int x, int blockSize){
	
	MPI_Request request;
	
	int z;
 
	int forw_rank = ((rank - 1) < gridSize*x)? gridSize*(x+1) - 1: rank - 1;

	// Send the block to the neigbouring node(forw_rank)
	MPI_Isend(matAlpha, blockSize*blockSize, MPI_INT, forw_rank, x, MPI_COMM_WORLD, &request);

	// Node from which the row block should be received
	int recv_rank = ((rank + 1) >= gridSize*(x+1))? gridSize*x: rank + 1;

	int *matTemp = malloc(sizeof(int)*blockSize*blockSize);

	// Receive the block from the neigbouring node(recv_rank)
	MPI_Recv(matTemp, blockSize*blockSize, MPI_INT, recv_rank, x, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	MPI_Wait(&request,MPI_STATUS_IGNORE);

	// Copy the data back to matAlpha
	for (z=0; z<blockSize*blockSize; z++)
		matAlpha[z] = matTemp[z];

	free(matTemp);
}

// Matrix column rotatation : Cannon's algorithm
void matrix_column_rotate(int rank, int gridSize, int* matBeta, int y, int blockSize){
	
	MPI_Request request;

	int z;
	
	// Node to which the row block should be forwarded
	int forw_rank = ((rank - gridSize) < 0)? gridSize * gridSize - (gridSize - rank): rank - gridSize;

	// Send the block to the neigbouring node(forw_rank)
	MPI_Isend(matBeta, blockSize*blockSize, MPI_INT, forw_rank, y, MPI_COMM_WORLD, &request);

	// Node from which the row block should be received
	int recv_rank = ((rank + gridSize) >= gridSize*gridSize)? (rank + gridSize) - gridSize*gridSize : rank + gridSize;

	int *matTemp = malloc(sizeof(int)*blockSize*blockSize);

	// Receive the block from the neigbouring node(recv_rank)
	MPI_Recv(matTemp, blockSize*blockSize, MPI_INT, recv_rank, y, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	MPI_Wait(&request,MPI_STATUS_IGNORE);

	// Copy the data back to matAlpha
	for (z=0; z<blockSize*blockSize; z++)
		matBeta[z] = matTemp[z];

	free(matTemp);
}

int main(){

	// Initialize the MPI world
	MPI_Init(NULL,NULL);
	
	int rank,num_procs;
	// Get the processor rank
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Get the MPI communicator world size
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	int *MatA, *MatB, *MatC, *MatResult;

	MatA = (int*)malloc(sizeof(int)* MATRIX_SIZE * MATRIX_SIZE);

	MatB = (int*)malloc(sizeof(int)* MATRIX_SIZE * MATRIX_SIZE);

	int i,j,k;

	int x,y,z;

	// Initializing matrix data
	for(x=0;x < MATRIX_SIZE * MATRIX_SIZE ;x++){
		MatA[x] = x%4;
		MatB[x] = x%4;
	}

	MatC = (int*)malloc(sizeof(int)* MATRIX_SIZE * MATRIX_SIZE);
	
	MatResult = (int*)malloc(sizeof(int)* MATRIX_SIZE * MATRIX_SIZE);
	
	int start_time, sum;
	
	// Debugging : Sequential execution at node 0
	if (!rank){
		printf("Matrix Multiplication size : %d , nodes : %d \n",MATRIX_SIZE, num_procs);

		start_time = MPI_Wtime();
		
		for (x = 0; x < MATRIX_SIZE; x++){
			for (y = 0; y < MATRIX_SIZE; y++){
				sum = 0;
				for (z=0; z < MATRIX_SIZE; z++)
					sum += MatA[x*MATRIX_SIZE+z]*MatB[z*MATRIX_SIZE+y];
				MatC[x*MATRIX_SIZE+y] = sum;
			}	
		}
	
		printf("Time taken (sequential): %f\n", MPI_Wtime() - start_time);

		// // Printing separately as it should not affect the run time 
		// for (x = 0; x < MATRIX_SIZE; x++) { 
		// 	for (y = 0; y < MATRIX_SIZE; y++) {
		// 		printf(" %d ",MatC[x*MATRIX_SIZE + y]);
		// 	}
		// 	printf("\n");
		// }
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (!rank)
		// Distributed execution
		start_time = MPI_Wtime();

	MPI_Datatype block, blocktype;

	int gridSize = (int)sqrt(num_procs);

	int blockSize = (int)(MATRIX_SIZE/gridSize);

	// printf("gridSize:%d, blockSize:%d\n",gridSize, blockSize);

	// Start of the various blocks
	// int disp = {0, 1, 2, 3, 32, 33, 34, 35, 64, 65, 66, 67, 96, 97, 98, 99}

	// Dynamic set
	int disp[num_procs];

	k =0;
	// start of the various blocks
	for (x=0; x < gridSize; x++)
		for (y=0; y<gridSize; y++){
			disp[k++] = x * blockSize * gridSize + y;
		}

	// No of blocks per node
	// int scount = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	
	int scount[num_procs];
	
	for(x=0; x<num_procs ; x++)
		scount[x] = 1;

	// No of elements in each block
	// int ecount = {64,.. }

	int ecount[num_procs];

	for(x=0; x<num_procs ; x++)
		ecount[x] = MATRIX_SIZE*MATRIX_SIZE;

	// Type vector representing a grid
	MPI_Type_vector(blockSize, blockSize, blockSize * gridSize, MPI_INT, &block);
	
	MPI_Type_commit(&block); 
	
	MPI_Type_create_resized(block, 0, blockSize*sizeof(int), &blocktype);

	MPI_Type_commit(&blocktype);

	// Receiving blocks
	int *matAlpha = malloc(sizeof(int)*blockSize*blockSize);
	int *matBeta  = malloc(sizeof(int)*blockSize*blockSize);

	// Intermediate result block
	int *matCharlie = malloc(sizeof(int)*blockSize*blockSize);

	// Scatter blocks of Matrix A (checker board pattern)
	MPI_Scatterv(MatA, scount, disp, blocktype, matAlpha, blockSize*blockSize, MPI_INT, 0, MPI_COMM_WORLD);

	// Scatter blocks of Matrix B (checker board pattern)
	MPI_Scatterv(MatB, scount, disp, blocktype, matBeta, blockSize*blockSize, MPI_INT, 0, MPI_COMM_WORLD);

	// // For debugging
	// if (!rank){
	// 	printf("Mat Alpha in rank 0\n");

	// 	// Display the computed matrix through parallel code
	// 	for (x = 0; x < blockSize; x++) { 
	// 		for (y = 0; y < blockSize; y++) {
	// 			printf(" %d ",matAlpha[x*blockSize + y]);
	// 		}
	// 		printf("\n");
	// 	}

	// 	printf("Mat Beta in rank 0\n");

	// 	// Display the computed matrix through parallel code
	// 	for (x = 0; x < blockSize; x++) { 
	// 		for (y = 0; y < blockSize; y++) {
	// 			printf(" %d ",matBeta[x*blockSize + y]);
	// 		}
	// 		printf("\n");
	// 	}
	// }


	// -------------------------
	// Cannon's algorithm
	// -------------------------

	MPI_Status status;

	MPI_Request request;

	x = y = rank/gridSize;
	
	// ------------------------
	// Initial set up
	// ------------------------ 
	
	// MatAlpha row wise distribution
	// Initial set up: Rotate the blocks (blocks in row i , left by i times) so that every node has approriate set of datas to multiply
	while(y>0){
		
		matrix_row_rotate(rank, gridSize, matAlpha, x, blockSize);

		y--;
	}

	// MatBeta column wise distribution

	x = y = rank%gridSize;

	// Initial set up: Rotate the blocks (blocks in row i , left by i times) so that every node has approriate set of datas to multiply
	while(y>0){
		
		matrix_column_rotate(rank, gridSize, matBeta, y, blockSize);

		y--;
	}

	// Clear the result matrix array - matCharlie
	for (x=0; x< blockSize*blockSize; x++){
		matCharlie[x] = 0;
	}

	// ------------------------
	// Matrix multiply
	// ------------------------ 
	int loop = gridSize;

	x = rank/gridSize;
	y = rank%gridSize;
	while(loop > 0){
		// Move row blocks to the immediate neighbour
		// Node to which the row block should be forwarded
		matrix_row_rotate(rank, gridSize, matAlpha, x, blockSize);
		
		matrix_column_rotate(rank, gridSize, matBeta, y, blockSize);
		
		int k,l,m;
		for (k = 0; k < blockSize; k++)
			for (l = 0; l < blockSize; l++){
				sum = 0;
				for (m=0; m < blockSize; m++)
					sum += matAlpha[k*blockSize+m]*matBeta[m*blockSize+l];
				matCharlie[k*blockSize+l] += sum;
			}
		loop--;
	}

	// // For debugging
	// if (!rank){
	// 	printf("Mat Charlie in rank 0\n");

	// 	// Display the computed matrix through parallel code
	// 	for (x = 0; x < blockSize; x++) { 
	// 		for (y = 0; y < blockSize; y++) {
	// 			printf(" %d ",matCharlie[x*blockSize + y]);
	// 		}
	// 		printf("\n");
	// 	}
	// }

	// Gather blocks of matCharlie (checker board pattern)
	MPI_Gatherv(matCharlie, blockSize*blockSize, MPI_INT, MatResult, ecount, disp, blocktype, 0, MPI_COMM_WORLD);

	if (!rank){
		printf("Time taken (distributed): %f\n", MPI_Wtime() - start_time);

		// // Display the computed matrix through parallel code
		// for (x = 0; x < MATRIX_SIZE; x++) { 
		// 	for (y = 0; y < MATRIX_SIZE; y++) {
		// 		printf(" %d ",MatResult[x*MATRIX_SIZE + y]);
		// 	}
		// 	printf("\n");
		// }
	}

	MPI_Barrier(MPI_COMM_WORLD);
	
	// Release heap memory
	free(MatA);
	free(MatB);
	free(MatC);
	free(MatResult);	

	MPI_Finalize();

	return 0;
}
