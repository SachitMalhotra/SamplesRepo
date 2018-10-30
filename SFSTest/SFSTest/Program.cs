using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.SolverFoundation.Common;
using Microsoft.SolverFoundation.Solvers;

namespace SFSTest
{
    class Program
    {
        static void Main(string[] args)
        {
            SimplexSolver solver = new SimplexSolver();
            int savid, vzvid;
            solver.AddVariable("Saudi", out savid);
            solver.AddVariable("Ven", out vzvid);
            solver.SetBounds(savid, 0, 9000);
            solver.SetBounds(vzvid, 0, 6000);

            int c1rid, c2rid, c3rid, goalrid;
            solver.AddRow("c1", out c1rid);
            solver.AddRow("c2", out c2rid);
            solver.AddRow("c3", out c3rid);
            solver.AddRow("goal", out goalrid);

            // add coefficients to constraint rows
            solver.SetCoefficient(c1rid, savid, 0.3);
            solver.SetCoefficient(c1rid, vzvid, 0.4);
            solver.SetBounds(c1rid, 2000, Rational.PositiveInfinity);
            solver.SetCoefficient(c2rid, savid, 0.4);
            solver.SetCoefficient(c2rid, vzvid, 0.2);
            solver.SetBounds(c2rid, 1500, Rational.PositiveInfinity);
            solver.SetCoefficient(c3rid, savid, 0.2);
            solver.SetCoefficient(c3rid, vzvid, 0.3);
            solver.SetBounds(c3rid, 500, Rational.PositiveInfinity);

            // add objective (goal) to model and specify minimization (==true)
            solver.SetCoefficient(goalrid, savid, 20);
            solver.SetCoefficient(goalrid, vzvid, 15);
            solver.AddGoal(goalrid, 1, true);

            solver.Solve(new SimplexSolverParams());

            Console.WriteLine("SA {0}, VZ {1}, C1 {2}, C2 {3}, C3 {4}, Goal {5}",
                                solver.GetValue(savid).ToDouble(), solver.GetValue(vzvid).ToDouble(),
                                solver.GetValue(c1rid).ToDouble(), solver.GetValue(c2rid).ToDouble(),
                                 solver.GetValue(c3rid).ToDouble(), solver.GetValue(goalrid).ToDouble());

            Console.ReadLine();


        }
    }
}
