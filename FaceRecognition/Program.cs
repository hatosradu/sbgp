// See https://aka.ms/new-console-template for more information
using System.Reflection;
using Emgu.CV;
using Emgu.CV.Structure;
using FaceRecognition.SBGP;

string path = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!;
path = Path.Combine(path, @"Data\");

Image<Gray, byte> image = new Image<Gray, byte>(path +@"images.jpeg");
Matrix<int>? sgbp = StructuralBinaryGradientPatterns.ComputeStructuralLabel(image);

Image<Gray, Byte> img = sgbp.Mat.ToImage<Gray, Byte>();
img.Save(path + $"images_label{DateTime.Now.Ticks}.jpeg");

Console.WriteLine();