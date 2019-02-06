using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace UseIris
{
    class Program
    {
        static void Main(string[] args)
        {
            string filePath = @"D:/Users/Sachit/source/repos/SamplesRepo/IrisData/IrisData/iris-data/saved_model.model";
            var cpu = DeviceDescriptor.UseDefaultDevice();
            Console.WriteLine($"Hello from CNTK for {cpu.Type} only!");
            Variable input;


            Function f = Function.Load(filePath, cpu);
            Variable inp = f.Arguments.Single();
            NDShape iShape = inp.Shape;

            var iDataMap = new Dictionary<Variable, Value>();
            float[] a = new float[]  { 1.2f, 2.3f, 3f, 4f, 8f, 2f, 3f, 1f } ;

            var iVal = Value.CreateBatch(iShape, a, cpu);
            iDataMap.Add(inp, iVal);

            Variable outVar = f.Output;
            var outDataMap = new Dictionary<Variable, Value>();
            outDataMap.Add(outVar, null);

            f.Evaluate(iDataMap, outDataMap, cpu);
            var oval = outDataMap[outVar];
            var odata = oval.GetDenseData<float>(outVar);

            Console.WriteLine("");
        }
    }
}
