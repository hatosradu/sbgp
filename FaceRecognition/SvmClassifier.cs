using System.Xml;
using Emgu.CV;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace FaceRecognition;

public class SvmClassifier
{
    private readonly FaceRecognizer _recognizer;
    private List<string> _names;

    public SvmClassifier()
    {
        _names = new List<string>();
        _recognizer = new EigenFaceRecognizer(80, double.PositiveInfinity);
        LoadTrainingData();
    }

    public string Recognize(Image<Gray, byte> image)
    {
        FaceRecognizer.PredictionResult result = _recognizer.Predict(image);

        if (result.Label == -1)
        {
            return "unknown";
        }

        if (result.Distance > -1) return _names[result.Label];
        else return "Unknown";
    }

    private void LoadTrainingData()
    {
        List<Image<Gray, byte>> trainingImages = new List<Image<Gray, byte>>();

        int totalLabels = 0;
        _names = new List<string>();
        List<int> labels = new List<int>();

        FileStream filestream = File.OpenRead(Utilities.XML_PATH);
        long filelength = filestream.Length;
        byte[] xmlBytes = new byte[filelength];
        filestream.Read(xmlBytes, 0, (int)filelength);
        filestream.Close();

        using MemoryStream xmlStream = new MemoryStream(xmlBytes);
        using (XmlReader xmlreader = XmlTextReader.Create(xmlStream))
        {
            while (xmlreader.Read())
            {
                if (xmlreader.IsStartElement())
                {
                    switch (xmlreader.Name)
                    {
                        case "NAME":
                            if (xmlreader.Read())
                            {
                                labels.Add(_names.Count);
                                _names.Add(xmlreader.Value.Trim());
                                totalLabels += 1;
                            }
                            break;
                        case "FILE":
                            if (xmlreader.Read())
                            {
                                trainingImages.Add(new Image<Gray, byte>(xmlreader.Value.Trim()));
                            }
                            break;
                    }
                }
            }
        }

        if (trainingImages.ToArray().Length != 0)
        {
            VectorOfMat matImages = new VectorOfMat();
            matImages.Push(trainingImages.ToArray());

            VectorOfInt matLabels = new VectorOfInt();
            matLabels.Push(labels.ToArray());

            _recognizer.Train(matImages, matLabels);
        }
    }
}
