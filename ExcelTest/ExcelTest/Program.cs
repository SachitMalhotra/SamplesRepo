using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OfficeOpenXml;

namespace ExcelTest
{
    class Program
    {
        static void Main(string[] args)
        {
            using (ExcelPackage excel = new ExcelPackage())
            {
                excel.Workbook.Worksheets.Add("Worksheet1");
                excel.Workbook.Worksheets.Add("Worksheet2");
                excel.Workbook.Worksheets.Add("Worksheet3");

                var wksht = excel.Workbook.Worksheets["Worksheet1"];
                List<string[]> headerRow = new List<string[]>() { new string[] { "ID", "First Name", "Last Name", "DOB" } };
                string headerRange = "A1:D1";
                ExcelRange range = wksht.Cells[headerRange];
                wksht.Names.Add("TestName", range);
                wksht.Cells[headerRange].LoadFromArrays(headerRow);

                System.IO.FileInfo ef = new System.IO.FileInfo(@"C:\Users\Sachit\Desktop\prog.xlsx");
                excel.SaveAs(ef);
            }
        }
    }
}
