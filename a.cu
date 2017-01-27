#include <iostream>
#include <stdio.h>
#include <climits>
#include <ctime>

void funcCPU(){
	for (int a = 0; a < ULONG_MAX; a++){
		for (int b = 0; b < ULONG_MAX; b++){ 
			for (int c = 0; c < ULONG_MAX; c++){
				for (int d = 0; d < ULONG_MAX; d++){
					for (int e = 0; e < ULONG_MAX; e++){
				
					}
				}
			}
		}
	}
}

__global__ void kernel( void (*func) (void) ){	func();	}

void LinearCPU();
void OffLoadGPU();

int main(){

	int numDevs;
	cudaGetDeviceCount(&numDevs);
	std::cout << "Number of devices found: " << numDevs << std::endl;
	std::cout << std::endl;

//	LinearCPU();	
	OffLoadGPU();
	return 1;
}


void LinearCPU(){

	clock_t time = clock();
	std::cout << "Beginning LinearCPU() ..." << std::endl;
	
	funcCPU();	
	
	time = clock() - time;
	std::cout << "LinearCPU() runtime:\t" << time / (double) CLOCKS_PER_SEC << std::endl;
	std::cout << "\n" << std::endl; 
}


void OffLoadGPU(){
	
	clock_t time = clock();
	std::cout << "Beginning OffLoadGPU() ..." << std::endl;

	// 1 - # thread blocks in the grid
	// 2 - # threads in a thread block
	// <<< blocks, threads >>> 
	kernel<<< (ULONG_MAX + 255) / 256, 256 >>>(funcCPU);	
	
	time = clock() - time;
	std::cout << "OffLoadGPU() runtime:\t" << time / (double) CLOCKS_PER_SEC << std::endl;
	std::cout << "\n" << std::endl; 
}
