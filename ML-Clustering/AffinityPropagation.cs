using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.Statistics;
using System.IO;
using MathNet.Numerics.LinearAlgebra.Double;

namespace ML_Clustering
{
    class AffinityPropagation
    {
        private Matrix<double> S;
        private Matrix<double> R;
        private Matrix<double> A;
        public AffinityPropagation(Matrix<double> data, bool simMatrix = false)
        {

       
            if (simMatrix)
                S = data;
            else
                S = GetSimilarityMatrix(data);
            R = S.Multiply(0.000000000001);
            A = S.Multiply(0.000000000001);

        }
        private static double Similarity(Vector<double> xi, Vector<double> xk)
        {
            return -(double)((xi - xk).L2Norm());
        }
        public static Matrix<double> GetSimilarityMatrix(Matrix<double> data)
        {
            Matrix<double> similarities = Matrix<double>.Build.Sparse(data.RowCount, data.RowCount);
            List<double> allSimilarities = new List<double>();
            for (int i = 0; i < data.RowCount; i++)
            {
                for (int j = 0; j < data.RowCount; j++)
                {
                    var sim = Similarity(data.Row(i), data.Row(j)); ;
                    allSimilarities.Add(sim);
                    similarities[i, j] = sim;
                }
            }
            var min = allSimilarities.Min();
            similarities.SetDiagonal(Vector<double>.Build.Dense(similarities.RowCount, min));

            return similarities;
        }
        public void IterateR(double damping = 0.7)
        {
            List<string> outInfo = new List<string>();
            Random rnd = new Random();
            double csum = 0;
            for (int i = 0; i < S.RowCount; i++)
            {
                var curRow = S.Row(i);
                double firstMax = double.MinValue, secondMax = double.MinValue;
                int firstMaxInd = -1;
                foreach (var j in curRow.EnumerateIndexed(Zeros.AllowSkip))
                {
                    var v = j.Item2 + A[j.Item1, i];
                    if (v > firstMax)
                    {
                        firstMaxInd = j.Item1;
                        secondMax = firstMax;
                        firstMax = v;
                    }
                    else
                        if (v > secondMax)
                        secondMax = v;
                }
                foreach (var j in curRow.EnumerateIndexed(Zeros.AllowSkip))
                {
                    //var max = j.Item1 == firstMaxInd ? secondMax : firstMax;
                    var s = S[i, j.Item1];
                    double newVal;
                    if (j.Item1 != firstMaxInd)
                    {
                        newVal = s - firstMax;
                    }
                    else
                    {
                        newVal  = s - secondMax; 

                    }
                    R[i, j.Item1] = R[i, j.Item1] * damping + (1 - damping) * newVal;
                    csum += Math.Abs(R[i, j.Item1]);
                }

                //49208.600001303639
                //49208.599999844904
            }
        }
        public void IterateA(double damping = 0.7)
        {
            double csum = 0;
            var Rt = R.Transpose();
            for(int i = 0; i<S.RowCount; i++)
            {
                var curRow = Rt.Row(i);
                double colSum = 0;
                foreach (var j in curRow.EnumerateIndexed(Zeros.AllowSkip))
                {
                    if (j.Item1 != i)
                        colSum += Math.Max(0, j.Item2);
                }
                double rkk = curRow[i];
                foreach (var j in curRow.EnumerateIndexed(Zeros.AllowSkip))
                {
                    if (j.Item1 != i)
                    {
                        A[i, j.Item1] = damping * A[i, j.Item1] + (1 - damping) * Math.Min(0, rkk + colSum - Math.Max(0, j.Item2));
                        csum += Math.Abs(A[i, j.Item1]);
                    }
                }
                A[i, i] = damping * A[i, i] + (1 - damping) * colSum;
                csum += Math.Abs(A[i, i]);
                

                
            }
        }

        //68340.541378626949
        //68340.541399342750


        public List<int> GetExemplars()
        {
            List<int> exemplars = new List<int>();
            var sum = A.Transpose() + R;
            for(int i = 0; i<S.RowCount; i++)
            {
                var ind = sum.Row(i).MaximumIndex();


                exemplars.Add(ind);


            }
          
            return exemplars;
        }

    }
}
