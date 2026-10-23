#include <iostream>
using namespace std;

void matmul_cpu(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
);

int main(){
    float A1[] = {1, 2, 3, 4, 5, 6};
    float B1[] = {7, 8, 9, 10, 11, 12};
    float expected1[] = {58, 64, 139, 154};
    float C1[4] = {};

    matmul_cpu(A1, B1, C1, 2, 2, 3);

    for(int i = 0; i < 4; i++){
        if(C1[i] != expected1[i]){
            cout << "FAIL" << endl;
            cout << "计算可能错误" << endl;
            return 1;
        }
    }

    float A2[] = {1, 2, 3, 4};
    float B2[] = {4, 2, 2, 3, 1, 5};
    float expected2[] = {10,4,12,24,10,26};
    float C2[6] = {};

    matmul_cpu(A2, B2, C2, 2, 3, 2);

    for(int i = 0; i < 6; i++){
        if(C2[i] != expected2[i]){
            cout << "FAIL" << endl;
            cout << "索引可能错误" << endl;
            return 1;
        }
    }    

    cout << "PASS\n";
    return 0;
}
