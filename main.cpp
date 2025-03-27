#include <iostream>
#include <iomanip>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

/* Funzione che calcola la fattorizzazione PA = LU della matrice A */ 
VectorXd palu_decomposition(MatrixXd A, VectorXd b)
{
    FullPivLU<MatrixXd> lu(A);
    return lu.solve(b);
}

/*  
    Funzione che calcola la fattorizzazione A = QR della matrice A.alignas
    La matrice A è di piccole dimensionioni, quindi ho usato FullPivHouseholderQR così da 
    ottenere un risultato più accurato. 
    Se la matrice fosse stata più grande FullPivHouseholderQR avrebbe appesantito troppo il 
    programma
*/ 
VectorXd qr_decomposition(MatrixXd A, VectorXd b)
{
    FullPivHouseholderQR<MatrixXd> qr(A);
    return qr.solve(b);
}

/* Funzione che calcola l'errore relatvo */
double relative_error(Vector2d solution)
{
    Vector2d real_solution;
    real_solution << -1.0e+00, -1.0e+00;
    return (real_solution-solution).norm()/real_solution.norm();
}

/* Funzione che calcola la soluzione dei sistemi e stampa i risultati */
void print_systems(const Matrix2d& A, const Vector2d& b) // Passaggio per riferimento costante
{
    cout << "\nSe la matrice A è:\nA = "<< scientific << setprecision(15)<< A;
    cout << "\n\ne il vettore b è:\n" << "b = " << b << endl;

    Vector2d solPALU = palu_decomposition(A, b);
    double errPALU = relative_error(solPALU);
    cout << "\nallora la soluzione del sistema lineare Ax = b con la fattorizzazione PA = LU è:\nx =  "<< solPALU << "\n\ncon un errore relativo pari a:\nerr = "<< errPALU <<endl;

    Vector2d solQR = qr_decomposition(A, b);
    double errQR = relative_error(solQR);
    cout << "\nmentre la soluzione del sistema lineare Ax = b con la fattorizzazione A = QR è:\nx =  "<< solQR << "\n\ncon un errore relativo pari a:\nerr = "<< errQR <<"\n\n\n"<<endl;
}


int main()
{
    Matrix2d M1;
    Vector2d b1;
    M1 << 5.547001962252291e-01, -3.770900990025203e-02,
            8.320502943378437e-01, -9.992887623566787e-01;
    
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;


    Matrix2d M2;
    Vector2d b2;
    M2 << 5.547001962252291e-01, -5.540607316466765e-01,
            8.320502943378437e-01, -8.324762492991313e-01;

    b2 << -6.394645785530173e-04, 4.259549612877223e-04;


    Matrix2d M3;
    Vector2d b3;
    M3 << 5.547001962252291e-01, -5.547001955851905e-01,
            8.320502943378437e-01, -8.320502947645361e-01;

    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    print_systems(M1, b1);
    print_systems(M2, b2);
    print_systems(M3, b3);

    return 0;
}
