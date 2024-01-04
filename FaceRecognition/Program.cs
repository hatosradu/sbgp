using System.Text;
using System.Xml;
using Emgu.CV;
using Emgu.CV.Structure;
using FaceRecognition;
using FaceRecognition.SBGP;

//Add new data to training
//For adding new trainging data, the images must have following format: `{subject_name} {xxxx}`
foreach (string file in Directory.EnumerateFiles(Utilities.APP_PATH + "NewData"))
{
    FileInfo fileInfo = new FileInfo(file);

    string subjectName = fileInfo.Name.Split(' ')[0];

    Image<Gray, byte> newImage = new Image<Gray, byte>(file);
    Matrix<int>? newImageSgbp = StructuralBinaryGradientPatterns.ComputeStructuralLabel(newImage);
    Image<Gray, Byte> newImageGray = newImageSgbp.Mat.ToImage<Gray, Byte>();
    SaveTrainingData(newImageGray, subjectName);
    File.Delete(fileInfo.FullName);
}

SvmClassifier classifier = new SvmClassifier();
foreach (string file in Directory.EnumerateFiles(Utilities.APP_PATH + "Recognize"))
{
    string imageToClassifyPath = file;
    Console.WriteLine("Subject: " + file);
    Image<Gray, byte> imageToClassify = new Image<Gray, byte>(imageToClassifyPath);
    string reesult = classifier.Recognize(imageToClassify);
    Console.WriteLine("Subject found: " + reesult);
}

Console.ReadLine();

static void SaveTrainingData(Image<Gray, Byte> image, string name)
{
    try
    {
        string faceName = "face_" + name + "_" + DateTime.UtcNow.Ticks + ".jpg";
        string trainedFacesPath = Path.Combine(Utilities.APP_PATH, "TrainedFaces/");
        if (!Directory.Exists(trainedFacesPath))
        {
            Directory.CreateDirectory(trainedFacesPath);
        }

        image.Save(trainedFacesPath + faceName);
        SaveImageLabel(name, trainedFacesPath + faceName);
    }
    catch (Exception ex)
    {
        Console.WriteLine("Failed to save training data for subject: {0}\n{1}", name, ex.Message);
    }
}

static void SaveImageLabel(string subjectName, string imagePath)
{
    string labelsPath = Path.Combine(Utilities.APP_PATH, "TrainedFaces/TrainedLabels.xml");
    if (!File.Exists(labelsPath))
    {
        using FileStream stream = File.OpenWrite(labelsPath);
        using (XmlWriter writer = XmlWriter.Create(stream, new XmlWriterSettings()
        {
            Encoding = Encoding.UTF8
        }))
        {
            writer.WriteStartDocument();
            writer.WriteStartElement("Faces_For_Training");

            writer.WriteStartElement("FACE");
            writer.WriteElementString("NAME", subjectName);
            writer.WriteElementString("FILE", imagePath);
            writer.WriteEndElement();

            writer.WriteEndElement();
            writer.WriteEndDocument();
        }
    }
    else
    {
        XmlDocument xmlDoc = new XmlDocument();
        xmlDoc.Load(labelsPath);

        XmlElement root = xmlDoc.DocumentElement!;

        XmlElement face_D = xmlDoc.CreateElement("FACE");
        XmlElement name_D = xmlDoc.CreateElement("NAME");
        XmlElement file_D = xmlDoc.CreateElement("FILE");

        name_D.InnerText = subjectName;
        file_D.InnerText = imagePath;

        face_D.AppendChild(name_D);
        face_D.AppendChild(file_D);

        root.AppendChild(face_D);
        xmlDoc.Save(labelsPath);
    }
}
