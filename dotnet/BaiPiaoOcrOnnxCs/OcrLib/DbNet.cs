using ClipperLib;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace OcrLiteLib
{
    class DbNet
    {
        private readonly float[] MeanValues = { 0.485F * 255F, 0.456F * 255F, 0.406F * 255F };
        private readonly float[] NormValues = { 1.0F / 0.229F / 255.0F, 1.0F / 0.224F / 255.0F, 1.0F / 0.225F / 255.0F };

        private InferenceSession dbNet;

        private List<string> inputNames;

        public DbNet() { }

        ~DbNet()
        {
            dbNet.Dispose();
        }

        public void InitModel(string path, int numThread)
        {
            try
            {
                SessionOptions op = new SessionOptions();
                op.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
                op.InterOpNumThreads = numThread;
                op.IntraOpNumThreads = numThread;
                dbNet = new InferenceSession(path, op);
                inputNames = dbNet.InputMetadata.Keys.ToList();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message + ex.StackTrace);
                throw ex;
            }
        }

        public List<TextBox> GetTextBoxes(Mat src, ScaleParam scale, float boxScoreThresh, float boxThresh, float unClipRatio)
        {
            Mat srcResize = new Mat();
            CvInvoke.Resize(src, srcResize, new Size(scale.DstWidth, scale.DstHeight));
            Tensor<float> inputTensors = OcrUtils.SubstractMeanNormalize(srcResize, MeanValues, NormValues);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputNames[0], inputTensors)
            };
            try
            {
                using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = dbNet.Run(inputs))
                {
                    var resultsArray = results.ToArray();
                    Console.WriteLine(resultsArray);
                    var textBoxes = GetTextBoxes(resultsArray, srcResize.Rows, srcResize.Cols, scale, boxScoreThresh, boxThresh, unClipRatio);
                    return textBoxes;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message + ex.StackTrace);
            }
            return null;
        }

        private static List<TextBox> GetTextBoxes(DisposableNamedOnnxValue[] outputTensor, int rows, int cols, ScaleParam s, float boxScoreThresh, float boxThresh, float unClipRatio)
        {
            float maxSideThresh = 3.0f;//长边门限
            List<TextBox> rsBoxes = new List<TextBox>();
            //-----Data preparation-----
            float[] predData = outputTensor[0].AsEnumerable<float>().ToArray();
            List<byte> cbufData = new List<byte>();
            foreach (float data in predData)
            {
                var val = data * 255;
                cbufData.Add(Convert.ToByte(val));
            }
            Mat predMat = new Mat(rows, cols, DepthType.Cv32F, 1);
            predMat.SetTo(predData);

            Mat cbufMat = new Mat(rows, cols, DepthType.Cv8U, 1);
            cbufMat.SetTo(cbufData.ToArray());

            //-----boxThresh-----
            Mat thresholdMat = new Mat();
            CvInvoke.Threshold(cbufMat, thresholdMat, boxThresh * 255.0, 255.0, ThresholdType.Binary);

            //-----dilate-----
            Mat dilateMat = new Mat();
            Mat dilateElement = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(2, 2), new Point(-1, -1));
            CvInvoke.Dilate(thresholdMat, dilateMat, dilateElement, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(128, 128, 128));

            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();

            CvInvoke.FindContours(dilateMat, contours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);

            for (int i = 0; i < contours.Size; i++)
            {
                if (contours[i].Size <= 2)
                {
                    continue;
                }
                float maxSide = 0;
                List<PointF> minBox = GetMiniBox(contours[i], out maxSide);
                if (maxSide < maxSideThresh)
                {
                    continue;
                }
                double score = GetScore(contours[i], predMat);
                if (score < boxScoreThresh)
                {
                    continue;
                }
                List<Point> clipBox = Unclip(minBox, unClipRatio);
                if (clipBox == null)
                {
                    continue;
                }

                List<PointF> clipMinBox = GetMiniBox(clipBox, out maxSide);
                if (maxSide < maxSideThresh + 2)
                {
                    continue;
                }
                List<Point> finalPoints = new List<Point>();
                foreach (var item in clipMinBox)
                {
                    int x = (int)(item.X / s.ScaleWidth);
                    int ptx = Math.Min(Math.Max(x, 0), s.SrcWidth);

                    int y = (int)(item.Y / s.ScaleHeight);
                    int pty = Math.Min(Math.Max(y, 0), s.SrcHeight);
                    Point dstPt = new Point(ptx, pty);
                    finalPoints.Add(dstPt);
                }

                TextBox textBox = new TextBox();
                textBox.Score = (float)score;
                textBox.Points = finalPoints;
                rsBoxes.Add(textBox);
            }
            rsBoxes.Reverse();
            return rsBoxes;
        }

        private static List<PointF> GetMiniBox(List<Point> contours, out float minEdgeSize)
        {
            VectorOfPoint vop = new VectorOfPoint();
            vop.Push(contours.ToArray<Point>());
            return GetMiniBox(vop, out minEdgeSize);
        }

        private static List<PointF> GetMiniBox(VectorOfPoint contours, out float minEdgeSize)
        {
            List<PointF> box = new List<PointF>();
            RotatedRect rrect = CvInvoke.MinAreaRect(contours);
            PointF[] points = CvInvoke.BoxPoints(rrect);
            minEdgeSize = Math.Min(rrect.Size.Width, rrect.Size.Height);

            List<PointF> thePoints = new List<PointF>(points);
            thePoints.Sort(CompareByX);

            int index_1 = 0, index_2 = 1, index_3 = 2, index_4 = 3;
            if (thePoints[1].Y > thePoints[0].Y)
            {
                index_1 = 0;
                index_4 = 1;
            }
            else
            {
                index_1 = 1;
                index_4 = 0;
            }

            if (thePoints[3].Y > thePoints[2].Y)
            {
                index_2 = 2;
                index_3 = 3;
            }
            else
            {
                index_2 = 3;
                index_3 = 2;
            }

            box.Add(thePoints[index_1]);
            box.Add(thePoints[index_2]);
            box.Add(thePoints[index_3]);
            box.Add(thePoints[index_4]);

            return box;
        }

        public static int CompareByX(PointF left, PointF right)
        {
            if (left == null && right == null)
            {
                return 1;
            }

            if (left == null)
            {
                return 0;
            }

            if (right == null)
            {
                return 1;
            }

            if (left.X > right.X)
            {
                return 1;
            }

            if (left.X == right.X)
            {
                return 0;
            }

            return -1;
        }

        private static double GetScore(VectorOfPoint contours, Mat fMapMat)
        {
            short xmin = 9999;
            short xmax = 0;
            short ymin = 9999;
            short ymax = 0;

            try
            {
                foreach (Point point in contours.ToArray())
                {
                    if (point.X < xmin)
                    {
                        //var xx = nd[point.X];
                        xmin = (short)point.X;
                    }

                    if (point.X > xmax)
                    {
                        xmax = (short)point.X;
                    }

                    if (point.Y < ymin)
                    {
                        ymin = (short)point.Y;
                    }

                    if (point.Y > ymax)
                    {
                        ymax = (short)point.Y;
                    }
                }

                int roiWidth = xmax - xmin + 1;
                int roiHeight = ymax - ymin + 1;

                Image<Gray, float> bitmap = fMapMat.ToImage<Gray, float>();
                Image<Gray, float> roiBitmap = new Image<Gray, float>(roiWidth, roiHeight);
                float[,,] dataFloat = bitmap.Data;
                float[,,] data = roiBitmap.Data;

                for (int j = ymin; j < ymin + roiHeight; j++)
                {
                    for (int i = xmin; i < xmin + roiWidth; i++)
                    {
                        try
                        {
                            data[j - ymin, i - xmin, 0] = dataFloat[j, i, 0];
                        }
                        catch (Exception ex2)
                        {
                            Console.WriteLine(ex2.Message);
                        }
                    }
                }

                Mat mask = Mat.Zeros(roiHeight, roiWidth, DepthType.Cv8U, 1);
                List<Point> pts = new List<Point>();
                foreach (Point point in contours.ToArray())
                {
                    pts.Add(new Point(point.X - xmin, point.Y - ymin));
                }

                using (VectorOfPoint vp = new VectorOfPoint(pts.ToArray<Point>()))
                using (VectorOfVectorOfPoint vvp = new VectorOfVectorOfPoint(vp))
                {
                    CvInvoke.FillPoly(mask, vvp, new MCvScalar(1));
                }

                return CvInvoke.Mean(roiBitmap, mask).V0;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message + ex.StackTrace);
            }

            return 0;
        }

        private static List<Point> Unclip(List<PointF> box, float unclip_ratio)
        {
            RotatedRect clipRect = CvInvoke.MinAreaRect(box.ToArray());
            if (clipRect.Size.Height < 1.001 && clipRect.Size.Width < 1.001)
            {
                return null;
            }

            List<IntPoint> theCliperPts = new List<IntPoint>();
            foreach (PointF pt in box)
            {
                IntPoint a1 = new IntPoint((int)pt.X, (int)pt.Y);
                theCliperPts.Add(a1);
            }

            float area = Math.Abs(SignedPolygonArea(box.ToArray<PointF>()));
            double length = LengthOfPoints(box);
            double distance = area * unclip_ratio / length;

            ClipperOffset co = new ClipperOffset();
            co.AddPath(theCliperPts, JoinType.jtRound, EndType.etClosedPolygon);
            List<List<IntPoint>> solution = new List<List<IntPoint>>();
            co.Execute(ref solution, distance);
            if (solution.Count == 0)
            {
                return null;
            }

            List<Point> retPts = new List<Point>();
            foreach (IntPoint ip in solution[0])
            {
                retPts.Add(new Point((int)ip.X, (int)ip.Y));
            }

            return retPts;
        }

        private static float SignedPolygonArea(PointF[] Points)
        {
            // Add the first point to the end.
            int num_points = Points.Length;
            PointF[] pts = new PointF[num_points + 1];
            Points.CopyTo(pts, 0);
            pts[num_points] = Points[0];

            // Get the areas.
            float area = 0;
            for (int i = 0; i < num_points; i++)
            {
                area +=
                    (pts[i + 1].X - pts[i].X) *
                    (pts[i + 1].Y + pts[i].Y) / 2;
            }

            return area;
        }

        private static double LengthOfPoints(List<PointF> box)
        {
            double length = 0;

            PointF pt = box[0];
            double x0 = pt.X;
            double y0 = pt.Y;
            double x1 = 0, y1 = 0, dx = 0, dy = 0;
            box.Add(pt);

            int count = box.Count;
            for (int idx = 1; idx < count; idx++)
            {
                PointF pts = box[idx];
                x1 = pts.X;
                y1 = pts.Y;
                dx = x1 - x0;
                dy = y1 - y0;

                length += Math.Sqrt(dx * dx + dy * dy);

                x0 = x1;
                y0 = y1;
            }

            box.RemoveAt(count - 1);
            return length;
        }

    }
}

