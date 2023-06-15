using Emgu.CV;
using Emgu.CV.Structure;

namespace FaceRecognition.SBGP;

public class StructuralBinaryGradientPatterns
{
    public static Matrix<int> ComputeDescriptor(Image<Gray, byte> image)
    {
        Matrix<int> sbgp = new Matrix<int>(image.Height, image.Width);
        int neigbors = 8;
        int radius = 1;

        byte[] principalBinaryNumbers = new byte[neigbors / 2 + 1];

        for (int i = radius; i < image.Height - radius; i++)
        {
            for (int j = radius; j < image.Width - radius; j++)
            {
                int index = 1;

                for (int n1 = -radius; n1 <= radius; n1++)
                {
                    int x = image.Data[i + n1, j + radius, 0];
                    int y = image.Data[i - n1, j - radius, 0];

                    if (x - y >= 0)
                    {
                        principalBinaryNumbers[index] = 1;
                    }
                    else
                    {
                        principalBinaryNumbers[index] = 0;
                    }

                    index++;
                }

                for (int n2 = -(radius - 1); n2 <= radius - 1; n2++)
                {
                    int pixelIntensity_1 = image.Data[i + radius, j - n2, 0];
                    int pixelIntensity_2 = image.Data[i - radius, j + n2, 0];

                    if (pixelIntensity_1 - pixelIntensity_2 >= 0)
                    {
                        principalBinaryNumbers[index] = 1;
                    }
                    else
                    {
                        principalBinaryNumbers[index] = 0;
                    }

                    index++;
                }

                int structuralLabel = 0;
                for (int l = 0; l < principalBinaryNumbers.Length; l++)
                {
                    structuralLabel += (int)(Math.Pow(2, l - 1) * principalBinaryNumbers[l]);
                }

                sbgp.Data[i, j] = structuralLabel;
            }
        }

        return sbgp;
    }
}
