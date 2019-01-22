using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML_Clustering
{
    struct Edge
    {
        public int Source { get; set; }
        public int Destination { get; set; }
        public double Similarity { get; set; }
        public double Responsibility { get; set; }
        public double Availability { get; set; }

    }

}
