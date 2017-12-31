#include <iostream>
#include <stdio.h>
#include <climits>
#include <ctime>

#define BLOCKS 16
#define THREADS 512

unsigned long long int BILLION     = 1000000000;
unsigned long long int TRILLION    = 1000000000000;
unsigned long long int QUADRILLION = 1000000000000000;
unsigned long long int LIMIT       = 10000 * QUADRILLION;

// Used to pass a CPU function as a param to the GPU
__global__ void kernel( void (*func) (void) ) {	func();	}


// Example 1 
/*#############################################################################
 * Compare execution time when running a CPU-defined function on the CPU vs 
 * passing  the same function as a param and running on the GPU 
#############################################################################*/
void Run_func_on_CPU();
void Run_func_on_GPU();
void funcCPU();	

inline void Example_1(bool CPU, bool GPU){
	if (CPU) Run_func_on_CPU();
	if (GPU) Run_func_on_GPU();
}


// Example 2 
/*#############################################################################
 * Compare execution time when running a CPU-defined function on the CPU vs 
 * running the same function as GPU defined and running on the GPU 
#############################################################################*/
void AlgoCPU();
__global__ 
void AlgoGPU(unsigned long long int DEV_LIMIT);
void Run_AlgoCPU();
void Run_AlgoGPU();

inline void Example_2(bool CPU, bool GPU){
	if (CPU) Run_AlgoCPU();
	if (GPU) Run_AlgoGPU();
}



//##############
// Example 1
//##############
// Executes CPU version of func() on CPU
void Run_func_on_CPU(){

	clock_t time = clock();
	std::cout << "Beginning Run_func_on_CPU() ..." << std::endl;
	
	funcCPU();	
	
	time = clock() - time;
	std::cout << "Run_func_on_CPU() runtime:\t" << time / (double) CLOCKS_PER_SEC << std::endl;
	std::cout << "\n" << std::endl; 
}

// Executes CPU version of func() on GPU
void Run_func_on_GPU(){

	clock_t time = clock();
	std::cout << "Beginning Run_func_on_GPU() ..." << std::endl;

	// 1 - # thread blocks in the grid
	// 2 - # threads in a thread block
	// <<< blocks, threads >>> 
	kernel<<< BLOCKS, THREADS >>>(funcCPU);	

	time = clock() - time;
	std::cout << "Run_func_on_GPU() runtime:\t" << time / (double) CLOCKS_PER_SEC << std::endl;
	std::cout << "\n" << std::endl; 
}

// func CPU version 
void funcCPU(){
	for (int a = 0; a < LIMIT; a++){
		for (int b = 0; b < LIMIT; b++){ 
			for (int c = 0; c < LIMIT; c++){
			}
		}
	}
}
// END Example 1
//##############


//##############
// Example 2 
//##############
// Run AlgoCPU()
void Run_AlgoCPU(){

	clock_t time = clock();
	std::cout << "Beginning AlgoCPU() with LIMIT = " << LIMIT << " ..." << std::endl;
	
	AlgoCPU();
	
	time = clock() - time;
	std::cout << "AlgoCPU() runtime:\t" << time / (double) CLOCKS_PER_SEC << std::endl;
	std::cout << "\n" << std::endl; 
}

// Run AlgoGPU()
void Run_AlgoGPU(){

	clock_t time = clock();
	std::cout << "Beginning AlgoGPU() with LIMIT = " << LIMIT << " ..." << std::endl;

	AlgoGPU<<< BLOCKS, THREADS >>>(LIMIT);
	
	time = clock() - time;
	std::cout << "AlgoGPU() runtime:\t" << time / (double) CLOCKS_PER_SEC << std::endl;
	std::cout << "\n" << std::endl; 
}

// Run an algorithm on the CPU (results do not matter; this is for time)
void AlgoCPU(){
		
	// This will overflow but that's not important
	long int sum = 0;

	for(int i = 0; i < LIMIT; i++){
		sum += (i * i) + 2 * (i * i * i) + i;
		for(int j = 0; j < 300; j++){
		
		}
		//std::cout << sum << std::endl;	
	}
}

// Run an algorithm on the GPU (results do not matter; this is for time)
__global__ void AlgoGPU(unsigned long long int DEV_LIMIT){

	// This will overflow but that's not important
	unsigned long long int sum = 0;
	
	for(unsigned long long int i = 0; i < DEV_LIMIT; i++){
		sum += (i * i) + 2 * (i * i * i) + i;
		for(int j = 0; j < 300; j++){
		
		}
	}
 }
// END Example 2
//##############
 


int main(){

	int numDevs;
	cudaGetDeviceCount(&numDevs);
	std::cout << "Number of devices found: " << numDevs << std::endl;
	std::cout << "CPU CLOCKS_PER_SEC: " << CLOCKS_PER_SEC << '\n' << std::endl;
	
	// Example_X(bool CPU, bool GPU);
	//Example_1(false, true);
	Example_2(false, true);
	
	return 1;
}


