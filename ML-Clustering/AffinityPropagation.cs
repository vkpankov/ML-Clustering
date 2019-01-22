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
        public AffinityPropagation(Matrix<double> data)
        {
            S = data;
            R = S.Multiply(0.000000000001);
            A = S.Multiply(0.000000000001);
        }
        
        public void IterateR(double damping = 0.9)
        {
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
                    var max = j.Item1 == firstMaxInd ? secondMax : firstMax;
                    var s = S[i, j.Item1];
                    R[i, j.Item1] = R[i, j.Item1] * damping + (1 - damping) * (s - max);
                }
            }
        }
        public void IterateA(double damping = 0.9)
        {
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
                foreach (var j in curRow.EnumerateIndexed(Zeros.AllowSkip))
                {
                    if (j.Item1 != i)
                    {
                        A[i, j.Item1] = damping * A[i, j.Item1] + (1 - damping) * Math.Min(0, curRow[i] + colSum - Math.Max(0, j.Item2));
                    }
                }
                A[i, i] = damping * A[i, i] + (1 - damping) * colSum;
            }
        }

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
