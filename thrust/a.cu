#include <iostream>
#include <time.h>
#include <list>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>


#define vector_size 2000000

using namespace std;


void Example_1(){
	int i;
	clock_t time;
	thrust::host_vector<int> vec1(vector_size);
	thrust::host_vector<int> vec2(vector_size);
	thrust::device_vector<int> vec4(vector_size);
	thrust::device_vector<int> fill(vector_size);

	thrust::fill(vec4.begin(), vec4.end(), -7);
	
	time = clock();
	// Populate some test values
	for(i = 0; i < vector_size; i++){
		vec1[i] = i;
		vec2[i] = i;
	}
	time = clock() - time;
	cout << "Set host vectors took " << time / CLOCKS_PER_SEC << " seconds.\n" << endl;
				

	time = clock();
	thrust::device_vector<int> vec3 = vec1;
	time = clock() - time;
	cout << "Setting vec3 = vec1 took " << time / CLOCKS_PER_SEC << " seconds.\n" << endl;


	time = clock();
	for(i = 0; i < vec1.size(); i++)
		vec1[i] = vec1[i] * 2;
	time = clock() - time;
	cout << "CPU: Settting vec1[i] = vec1[i] * 2 took " << time / CLOCKS_PER_SEC << " seconds.\n" << endl;
	
	time = clock();
	thrust::transform(vec4.begin(), vec4.end(), fill.begin(), thrust::negate<int>());
	time = clock() - time;
	cout << "thrust::transform negate took " << time / CLOCKS_PER_SEC << " seconds.\n" << endl;

}


typedef thrust::device_vector<int>::iterator	itrInt;
typedef thrust::device_vector<char>::iterator	itrChar;
typedef thrust::tuple<itrInt, itrChar>		itrTuple;
typedef thrust::zip_iterator<itrTuple> 		itrNLC;
void Example_2(){

	int i;
	int pair_size = 5;
	clock_t time;
	
	time = clock();
	thrust::device_vector<int> ID(pair_size);
	thrust::device_vector<char> neighbor(pair_size);

	ID[0] = 0; ID[1] = 1; ID[2] = 2; ID[3] = 3; ID[4] = 4;
	
	neighbor[0] = 'A';
	neighbor[1] = 'B';
	neighbor[2] = 'C';
	neighbor[3] = 'D';
	neighbor[4] = 'E';
 

	itrNLC first = thrust::make_zip_iterator( thrust::make_tuple(ID.begin(), neighbor.begin() ) );
	itrNLC last  = thrust::make_zip_iterator( thrust::make_tuple(ID.end(), neighbor.end() ) );

	for(i = 0; i < pair_size; i++)	
		cout <<  "ID: " << thrust::get<0>(first[i]) << "\tChar: " << thrust::get<1>(first[i]) << endl;	


	time = clock() - time;
	cout << "Execution time: " << time / CLOCKS_PER_SEC << " seconds." << endl;
}

typedef thrust::host_vector<int>::iterator				itrInt_2;
typedef thrust::host_vector<thrust::host_vector<int>>::iterator		itrVector;
typedef thrust::tuple<itrInt_2, itrVector>				itrTuple_2;
typedef thrust::zip_iterator<itrTuple_2> 				itrNLC_2;		// Use for NLC implementation
void Example_3(){

	int i;
	int pair_size = 5;
	int nSize = 5;		// max num of neighbors
	clock_t time;
	
	time = clock();
	thrust::host_vector<int> ID(pair_size);
	ID[0] = 0; ID[1] = 1; ID[2] = 2; ID[3] = 3; ID[4] = 4;

	thrust::host_vector<thrust::host_vector<int>> neighbors(pair_size);

	thrust::host_vector<int> vec1(nSize);
	thrust::host_vector<int> vec2(nSize);
	thrust::host_vector<int> vec3(nSize);
	thrust::host_vector<int> vec4(nSize);
	thrust::host_vector<int> vec5(nSize);

	for(i = 0; i < pair_size; i++){
		vec1[i] = i;
		vec2[i] = i;
		vec3[i] = i;
		vec4[i] = i;
		vec5[i] = i;
	}

	neighbors[0] = vec1;
	neighbors[1] = vec2;
	neighbors[2] = vec3;
	neighbors[3] = vec4;
	neighbors[4] = vec5;
	
	itrNLC_2 first = thrust::make_zip_iterator( thrust::make_tuple(ID.begin(), neighbors.begin() ) );
	itrNLC_2 last  = thrust::make_zip_iterator( thrust::make_tuple(ID.end()  , neighbors.end()   ) );

	// for each touple
	for(i = 0; i < pair_size; i++){	

		// print first element of touple
		cout <<  "ID: " << thrust::get<0>(first[i]) << "\tValues: ";

		// print second element of touple ( host_vector -> iterate through )		
		for(int j = 0; j < thrust::get<1>( first[i] ).size(); j++)
		 	cout << '\t' << thrust::get<1>( first[i] )[j];	
		cout << endl;
	}
	time = clock() - time;
	cout << "Execution time: " << time / CLOCKS_PER_SEC << " seconds." << endl;
}

int main(){
	Example_1();
	Example_2();
	Example_3();
	return 1;
}



