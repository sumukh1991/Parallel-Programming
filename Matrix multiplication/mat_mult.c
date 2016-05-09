/****
Author		: Halymysore Ravindra, Sumukh
Date		: 03/05/2016
Description	: A distributed memory program for Matrix multiplication using MPI library
****/

#include "stdio.h"
#include "mpi.h"
#include "stdlib.h"
#include "assert.h"

#define MATRIX_SIZE 40

int main(){

	MPI_Init(NULL,NULL);

	// Rank of the node , no of nodes in the process
	int rank, num_procs, forw_rank;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	
	int i, j, k, x, y, sum, element_count, position_count, count = (MATRIX_SIZE * MATRIX_SIZE)/ num_procs;

	// Pointer to the various matrix rows and/or columns
	int *mat_a = NULL, *mat_b = NULL, *mat_c = NULL;

	int *row_mult = NULL, *coulmn_mult = NULL, *result_mult = NULL, *result_mult_temp = NULL, *coulmn_mult_temp = NULL;

	double start_time;

	// To get the rank of the node that sent which was received along with length of the message and its type
	MPI_Status status;

	MPI_Request request;

	// Allocate space for each separate column/s of matrix B in each node 
	mat_b = (int *)malloc(sizeof(int) * count);
	
	assert(mat_b != NULL);
	// column_mult = (int *)malloc(sizeof(int) * count);

	// Intialize part of the Matrix B in each nodes in the MPI world
	// These hold the columns of the Matrix B
	//srand(1000+rank);
	for (i = 0; i < count; i++){
		mat_b[i] = rank + 10;
		//mat_b[i] = ((double)rand()/RAND_MAX) * 10;
	}
	//printf("Allocated memory and init mat b [%d]\n",rank);

	//MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 0)
		start_time = MPI_Wtime();

	
	row_mult = (int *)malloc(sizeof(int) * count);
		
	// Receive the Matrix A values in the nodes
	MPI_Irecv(row_mult, count, MPI_INT, 0, 1, MPI_COMM_WORLD, &request);

	// Initializing Matrix A in root node (rank = 0) as a sequence of linear array of integers
	if (rank == 0){
		// Allocate space for the matrix
		mat_a = (int *)malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);
		
		//srand(time(NULL));
		// Initialize the matrix
		for (i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++){
			//mat_a[i] = 1;
			mat_a[i] = ((double)rand()/RAND_MAX) * 10;
		}

		for (i = 0; i < num_procs; i++){
			// Distribute the values amongst the various nodes
			MPI_Send(mat_a + count * i, count, MPI_INT, i, 1, MPI_COMM_WORLD);	
			//printf("Distributing mat_a values starting from %d to node [%d]\n",mat_a[count*i],rank);			
		} 
	}
	
	MPI_Wait(&request,MPI_STATUS_IGNORE);

	//printf("Received mat_a values [%d] with value %d\n",rank,row_mult[0]);

	element_count = 0;
	position_count = rank;

	coulmn_mult = (int *)malloc(sizeof(int) * count);
	coulmn_mult = mat_b;

	result_mult = (int *)malloc(sizeof(int) * count);

	// Dummy allocation
	coulmn_mult_temp =(int *)malloc(sizeof(int) * count);

	// Iterate till all the elements of the result matrix is obtained
	// Multiply with the row_mult and the column_mult, and store the results locally
	// Once done, exchange the column_mult with other nodes and continue
	while(element_count < (MATRIX_SIZE/num_procs)){

		// Result is computed as block in each iteration starting from top left to bttom right, going from left to right
		// Later columns are exchanged between nodes and results are computed
		
		/*
		example : for 4X4 matrix with two processors, with order in which result elements are computed and saved
		Node 0
		-------------
		| 0 1 | 4 5 |
		| 2 3 | 6 7 |
		-------------

		result mat index: 0 1 4 5 2 3 6 7
		
		Node 1
		-------------
		| 4 5 | 0 1 |
		| 6 7 | 2 3 |
		-------------
		
		result mat index: 4 5 0 1 6 7 2 3

		Results are however stored in sequential manner, so as to make it easy when combining at the root node later
		*/
		

		for (i = 0; i < count/MATRIX_SIZE; i++) { 
			for (j = 0; j < count/MATRIX_SIZE; j++) {
				sum  = 0;
				for (k = 0; k < MATRIX_SIZE; k++) {
					sum += row_mult[i*MATRIX_SIZE+k] * coulmn_mult[j*MATRIX_SIZE+k];
				} 
				result_mult[MATRIX_SIZE*i + j + position_count*(count/MATRIX_SIZE)] = sum;
				//printf("[%d] result_mult[%d] = %d\n ",rank,MATRIX_SIZE*i + j + position_count*(count/MATRIX_SIZE),sum);
			}
		}
			
		//printf("[%d]:Computed result block %d\n",rank,element_count);		

		// Node to which teh column values should be forwarded
		forw_rank = ((rank - 1) < 0)? num_procs - 1: rank - 1;

		// Send the column values to the another node
		MPI_Isend(coulmn_mult, count, MPI_INT,forw_rank,2,MPI_COMM_WORLD, &request);

		// Node from which the column values should be received
		forw_rank = ((rank + 1) >= num_procs)? 0: rank + 1;

		// Receive the column vaulues from the another node
		MPI_Recv(coulmn_mult_temp, count, MPI_INT, forw_rank, 2,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Wait(&request,MPI_STATUS_IGNORE);

		for(i=0;i<count;i++)
			coulmn_mult[i] = coulmn_mult_temp[i];

		element_count++;

		// Get the starting index of the next block where the results needs to be saved
		position_count = ((position_count + 1)*(count/MATRIX_SIZE) >= MATRIX_SIZE)? 0: position_count + 1;

	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	// Send the results to the root node
	MPI_Isend(result_mult, count, MPI_INT, 0, 3, MPI_COMM_WORLD, &request);

	if(rank == 0){
		element_count = 0;

		mat_c = malloc(sizeof(int)*MATRIX_SIZE*MATRIX_SIZE);

		result_mult_temp = malloc(sizeof(int)*count);

		while(element_count < num_procs){

			MPI_Recv(result_mult_temp, count, MPI_INT, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, &status);
			
			printf("Received results from %d\n",status.MPI_SOURCE);
			
			if(status.MPI_SOURCE == 0)
				MPI_Wait(&request,MPI_STATUS_IGNORE);
			
			// Store the result in the proper place in the result matrix 
			for (i=0;i<count;i++)
				mat_c[status.MPI_SOURCE * count + i] = result_mult_temp[i];
			
			//printf("Aggregated results from %d\n",status.MPI_SOURCE);
			element_count++;
		}
		
		free(result_mult_temp);

		printf("Time taken (Distributed execution): %f\n", MPI_Wtime() - start_time);
		
		printf("\nDistributed matix product computed by segregating results at root node\n");

		// Display the computed matrix through parallel code
		for (x=0; x < MATRIX_SIZE;x++){
			for (y=0;y<MATRIX_SIZE;y++)
				printf(" %d ", mat_c[x*MATRIX_SIZE+y]);
			printf("\n");
		}
		
	}

	if(rank != 0)
		MPI_Wait(&request,MPI_STATUS_IGNORE);

	// Free all memory
	free(result_mult);
	free(row_mult);
	free(coulmn_mult);
	free(mat_b);

	// Check the result with the sequential multiply in root node 
	// by collecting all the values of Matrix B from each node
	// Sequential execution
	if(rank == 0){
		// Initialize a local matrix B for verification
		mat_b = (int *)malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);
		for (j=0;j<num_procs;j++) {
			//srand(1000+j);
			for (i = 0; i < count; i++)
				mat_b[i+j*count] = j + 10;
				//mat_b[i] = ((double)rand()/RAND_MAX) * 10;
		}
		
		printf("\nSequential matix product computed at root node.\n");

		// Display the computed matrix through parallel code
		for (i = 0; i < MATRIX_SIZE; i++) { 
			for (j = 0; j < MATRIX_SIZE; j++) {
				int sum  = 0;
				for (k = 0; k < MATRIX_SIZE; k++) {
					sum += mat_a[i*MATRIX_SIZE+k]*mat_b[j*MATRIX_SIZE+k];
				} 
				printf(" %d ",sum);
			}
			printf("\n");
		}

		free(mat_a);
		free(mat_b);
	}
	
	MPI_Barrier(MPI_COMM_WORLD);

	printf("[%d]: EOF reached\n",rank);
	MPI_Finalize();

	return(0);
}
