using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

using System.IO;
using System.Threading;
using System.Windows.Forms;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Statistics;

namespace ML_Clustering
{
    class Program
    {

        static Matrix<double> LoadData(int matrixSize, string fileName, string cacheFileName= "sparsematrix_cache.bin")
        {
            List<string> strData = File.ReadAllLines(fileName).ToList();
            Matrix<double> data = Matrix<double>.Build.Sparse(matrixSize, matrixSize);

            if (File.Exists(cacheFileName))
            {
                using (Stream inStream = new FileStream(cacheFileName, FileMode.Open, FileAccess.Read))
                {
                    IFormatter bf = new BinaryFormatter();
                    data = (SparseMatrix)bf.Deserialize(inStream);
                }
            }
            else
            {
                Random rnd = new Random();
                for (int i = 0; i < strData.Count; i++)
                {
                    string[] str = strData[i].Split('\t');
                    int row = Int32.Parse(str[0]);
                    int col = Int32.Parse(str[1]);
                    double rand = 1e-15 * rnd.NextDouble();
                    data[row, col] = 1 + rand;
                    data[row, row] = -1 + rand;
                }

                IFormatter formatter = new BinaryFormatter();
                Stream stream = new FileStream(@"sparsematrix_cache.bin", FileMode.Create, FileAccess.Write);

                formatter.Serialize(stream, data);
                stream.Close();
            }
            return data;
        }
        
        static List<int> GetClusters(Matrix<double> data, int itCount, int noUpdateCount)
        {
            AffinityPropagation ap = new AffinityPropagation(data, true);
            int prevExemplarsCount = 0;
            int n = 0;
            for (int i = 0; i < itCount && n < noUpdateCount; i++)
            {
                int startTime = Environment.TickCount;
                ap.IterateR(0.95);
                ap.IterateA(0.95);
                var exemplars = ap.GetExemplars().ToList();
                var exCount = exemplars.Distinct().Count();
                if (prevExemplarsCount == exCount)
                    n++;
                prevExemplarsCount = exCount;
                int endTime = Environment.TickCount;

                Console.WriteLine($"Iteration: {i}, exemplars count: {exCount}, time: {endTime - startTime}");
            }
            return ap.GetExemplars();
        }

        static void Main(string[] args)
        {

            Dictionary<int, HashSet<int>> userPlaces = new Dictionary<int, HashSet<int>>();

            var inputData = LoadData(196591, "Gowalla_edges.txt");
            var finalExemplars = GetClusters(inputData, 300, 200);

            //EVAL
            List<string> checkins = File.ReadAllLines(@"Gowalla_totalCheckins.txt").ToList();
            for (int i = 0; i < checkins.Count; i++)
            {
                string[] str = checkins[i].Split('\t');
                int user = Int32.Parse(str[0]);
                int place = Int32.Parse(str[4]);
                if (!userPlaces.ContainsKey(user))
                    userPlaces[user] = new HashSet<int>();
                userPlaces[user].Add(place);

            }

            Dictionary<int, List<int>> clusterRecommendedPlaces = new Dictionary<int, List<int>>();
            for (int i = 0; i < finalExemplars.Count; i++)
            {
                var userExemplar = finalExemplars[i];
                if (!clusterRecommendedPlaces.ContainsKey(userExemplar))
                    clusterRecommendedPlaces[userExemplar] = new List<int>();
                if (userPlaces.ContainsKey(i))
                    clusterRecommendedPlaces[userExemplar].AddRange(userPlaces[i].Take(userPlaces[i].Count));
            }

            for (int i = 0; i < clusterRecommendedPlaces.Count; i++)
            {
                var cel = clusterRecommendedPlaces.ElementAt(i);
                var q = from x in cel.Value
                        group x by x into g
                        orderby g.Count() descending
                        select g.Key;
                clusterRecommendedPlaces[cel.Key] = q.Take(10).ToList();
            }


            Vector<double> recommendPrec = Vector<double>.Build.Dense(userPlaces.Count);
            int k = 0;

            int totalRecommendedInCheckins = 0;
            int totalRecommended = 0;
            foreach (var i in userPlaces)
            {
                var recommendedPlacesForUserI = clusterRecommendedPlaces[finalExemplars[i.Key]];
                for (int j = 0; j < recommendedPlacesForUserI.Count; j++)
                {
                    if (userPlaces[i.Key].Contains(recommendedPlacesForUserI[j]))
                        totalRecommendedInCheckins++;
                    totalRecommended++;
                }
            }
            double prec = totalRecommendedInCheckins / (double)totalRecommended;
        
            Console.WriteLine($"Recommended: {totalRecommended}, in checkins: {totalRecommendedInCheckins}, {prec}%");

            Console.ReadKey();


        }
    }
}
