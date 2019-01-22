using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ML_Clustering
{
    public partial class PointVis : Form
    {

        public Matrix<double> Points2Dim { get; set; }

        public List<List<int>> DynExemplars { get; set; }


        public PointVis()
        {
            InitializeComponent();
        }

        public void UpdatePoints()
        {
            Color[] colors = new Color[] { Color.Blue, Color.Red, Color.Green };
            var knownColorsCount = Enum.GetValues(typeof(KnownColor)).Cast<KnownColor>().Count();
            foreach (var ExemplarIds in DynExemplars)
            {
                this.Invoke((MethodInvoker)delegate ()

                {
                    chart1.Series[1].Points.Clear();
                    chart1.Series[2].Points.Clear();

                    int curPoint = 0;
                    foreach(var i in ExemplarIds.Distinct())
                    {
                        chart1.Series[1].Points.AddXY(Points2Dim[i, 0], Points2Dim[i, 1]);
                        chart1.Series[1].Points[chart1.Series[1].Points.Count - 1].Color = colors[i % colors.Count()];
                    }
                    foreach (var i in ExemplarIds)
                    {


                        if (curPoint != i)
                        {
                            chart1.Series[0].Points[curPoint].Color = colors[i % colors.Count()];
                           // chart1.Series[2].Points.AddXY(Points2Dim[curPoint, 0], Points2Dim[curPoint, 1]);
                           // chart1.Series[2].Points.AddXY(Points2Dim[i, 0], Points2Dim[i, 1]);
                        }
                        curPoint++;
                    }

                });
     
             
                System.Threading.Thread.Sleep(50);
            }
        }

        private void PointVis_Load(object sender, EventArgs e)
        {
            for (int i = 0; i < Points2Dim.RowCount; i++)
            {
                    chart1.Series[0].Points.AddXY(Points2Dim[i, 0], Points2Dim[i, 1]);
            }
            new Thread(UpdatePoints).Start();
        }
    }
}
