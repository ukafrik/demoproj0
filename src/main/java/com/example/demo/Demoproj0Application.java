package com.example.demo;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class Demoproj0Application {

	public static void main(String[] args) {
		SpringApplication.run(Demoproj0Application.class, args);
	}
}

///////////New attempt:
package com.example.demo;

import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.LongAdder;

import javax.imageio.ImageIO;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageOutputStream;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.apache.tika.Tika;
import org.bytedeco.javacpp.indexer.UByteBufferIndexer;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.photo.Photo;

import com.recognition.software.jdeskew.ImageDeskew;
import com.recognition.software.jdeskew.ImageUtil;

import net.sourceforge.tess4j.ITesseract;
import net.sourceforge.tess4j.ITesseract.RenderedFormat;
import net.sourceforge.tess4j.Tesseract;
import net.sourceforge.tess4j.util.ImageHelper;

public class DemoTess4J {
	
	//Location of opencv binaries for Macos: https://github.com/opencv/opencv/archive/3.4.2.tar.gz
	
	//Note!!! Tess4J requires the ff to set for Tesseract to work!!!
	//-Djna.library.path=<Location of platform specific native dll/so>
	//-Dsun.java2d.cmm=sun.java2d.cmm.kcms.KcmsServiceProvider =>Improves Java 8/9/etc performance!!!!

	private static final int RUNS=10;
	public static void main(String[] args) {
		
//		Long timeNow = System.currentTimeMillis();
//		File[] files = new File("/home/ubuntumac1/DevArea/Workspace/simRel-2018-09/demoTessJ4/test-data/").listFiles();
//		System.out.println(String.format("Available procs: %d",  Runtime.getRuntime().availableProcessors()));
//		System.out.println(String.format("There are %d png files to OCR in paralled.",  files.length));
//		
//		final LongAdder totalErrs = new LongAdder();
//		
//		DemoTess4J demoP = new DemoTess4J();
//		for (int runIdx=1; runIdx<RUNS; runIdx++) {
//			final LongAdder errInCurRun = new LongAdder();
//			
//			Arrays.stream(files).parallel().forEach((file) -> {
//				try {
//					demoP.renderDocument(file.getAbsolutePath().replace("\\", "/"));
//				}
//				catch (Exception ex) {
//					errInCurRun.increment();
//				}
//			});
//			System.out.println(String.format("\tRun %d -> Errors: %d/%d",  runIdx, errInCurRun.intValue(), files.length));
//			totalErrs.add(errInCurRun.intValue());
//		}
//		int cntErrs = totalErrs.intValue();
//		System.out.println(String.format("Total Count-Errs: %d / Err Percentage: %.2f%%", 
//				cntErrs, cntErrs / (double) files.length *100 / RUNS));
//		
//		Long timeEnd = System.currentTimeMillis();
//		System.out.println("Elapsed Time [" + (timeEnd - timeNow) + "]");
		
		String inFile = "/home/ubuntumac1/DevArea/Workspace/simRel-2018-09/demoTessJ4/test-data/eurotext_deskew.png";
		DemoTess4J demo = new DemoTess4J();
		try {
			//demo.renderDocument(inFile);
			//demo.declareRectOfSkewArea(inFile);
			//demo.computeSkew(inFile);
			demo.boundingBox(inFile);
		} 
		catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private void renderDocument(String inFile) throws Exception {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		ITesseract instance = new Tesseract();
		instance.setDatapath("/home/ubuntumac1/DevArea/Workspace/simRel-2018-09/demoTessJ4/tessdata");
		
		String inFPathNoExt = inFile.substring(0, inFile.lastIndexOf("."));
		String inFName = inFPathNoExt.substring(inFPathNoExt.lastIndexOf("/") + 1);
		String outFile = inFile.substring(0, inFile.lastIndexOf("/") + 1) + "output/" + inFName;
		String outputImgFPath = outFile + ".png";
		
		try {
			BufferedImage srcBufImg = readImage(inFile);
			srcBufImg = this.imageToBinary(srcBufImg);
			Mat bufImgToMat = this.bufferedImagetoMat(srcBufImg);
			//bufImgToMat = this.removeBlkBorder(bufImgToMat);
			Mat enhancedImgMat = this.denoiseAndDetainEnhanceImage(bufImgToMat);
			BufferedImage mat2BufImage = this.matToBufferedImage(enhancedImgMat, ".png");
			BufferedImage deskewedImg = this.imageDeSkew(mat2BufImage);
			byte[] deskewedByteAryImg = this.imageWriter("png", deskewedImg);
			
			Mat result = Imgcodecs.imdecode(new MatOfByte(deskewedByteAryImg), Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
			//result = this.removeBlkBorder(result);
			//result = this.makeBlackTransparent(result);
			Imgcodecs.imwrite(outputImgFPath, result);
			
			//Convert created image to PDF...
			List<RenderedFormat> lstRendFmt = Arrays.asList(RenderedFormat.PDF);
			File pngFPath = new File(outputImgFPath);
			instance.createDocuments(pngFPath.getAbsolutePath(), outFile, lstRendFmt);		
		}
		catch(Exception ex) {
			ex.printStackTrace();
		}
	}
	
	private byte[] generateImgFromPDF(String inputPdfFPath, String format) throws IOException {
		ByteArrayOutputStream byteAryOS = null;
		
		PDDocument doc = null;
		try {
			doc = PDDocument.load(new File(inputPdfFPath));
			PDFRenderer pdfRendr = new PDFRenderer(doc);
			
			byteAryOS = new ByteArrayOutputStream();
			int pageCnt = doc.getNumberOfPages();
			
			byteAryOS = new ByteArrayOutputStream();
			ImageWriter writer = ImageIO.getImageWritersByFormatName(format).next();
			ImageOutputStream imgOS = ImageIO.createImageOutputStream(byteAryOS);

			for (int page=0; page < pageCnt; page++) {
				BufferedImage bufImg = pdfRendr.renderImageWithDPI(page, 300, ImageType.BINARY);
				
				RenderedImage rendImg = (RenderedImage) bufImg;
				byteAryOS = new ByteArrayOutputStream();
				//ImageWriter writer = ImageIO.getImageWritersByFormatName(format).next();
				//ImageOutputStream imgOS = ImageIO.createImageOutputStream(byteAryOS);
				
				writer.setOutput(imgOS);
				writer.write(rendImg);
			}
		}
		finally {
			doc.close();
			byteAryOS.flush();
			byteAryOS.close();
		}
		
		return byteAryOS.toByteArray();
	}
	
	/**

	 * Make the black background of a PNG-Bitmap-Image transparent.
	 * code based on example at http://j.mp/1uCxOV5
	 * @Param image png bitmap image
	 * @return output image
	 */

	private Mat makeBlackTransparent(Mat image) {
	    // convert image to matrix
	    Mat src = new Mat(image.width(), image.height(), CvType.CV_8UC4);

	    // init new matrices
	    Mat dst = new Mat(image.width(), image.height(), CvType.CV_8UC4);
	    Mat tmp = new Mat(image.width(), image.height(), CvType.CV_8UC4);
	    Mat alpha = new Mat(image.width(), image.height(), CvType.CV_8UC4);

	    // convert image to grayscale
	    Imgproc.cvtColor(src, tmp, Imgproc.COLOR_BGR2GRAY);

	    // threshold the image to create alpha channel with complete transparency in black background region and zero transparency in foreground object region.
	    Imgproc.threshold(tmp, alpha, 100, 255, Imgproc.THRESH_BINARY);

	    // split the original image into three single channel.
	    List<Mat> rgb = new ArrayList<Mat>(3);
	    Core.split(src, rgb);

	    // Create the final result by merging three single channel and alpha(BGRA order)
	    List<Mat> rgba = new ArrayList<Mat>(4);
	    rgba.add(rgb.get(0));
	    rgba.add(rgb.get(1));
	    rgba.add(rgb.get(2));
	    rgba.add(alpha);
	    Core.merge(rgba, dst);

	    // convert matrix to output bitmap
	    //Mat output = Bitmap.createBitmap(image.width(), image.height(), Bitmap.Config.ARGB_8888);
	    Mat output = new Mat(image.width(), image.height(), Imgproc.COLOR_BGR2GRAY);
		Imgproc.cvtColor(image, output, Imgproc.COLOR_BGR2GRAY);
	    return output;
	}
	
	private void edgeDetector(Mat srcMat) {
		Mat dstMat = new Mat();
		
		//Convert image to grey
		Imgproc.cvtColor(srcMat, dstMat, Imgproc.COLOR_BGR2GRAY);
		
		//Mat detectedEdges = new Mat();
		Imgproc.blur(dstMat, dstMat, new Size(3, 3));
		
		//Apply Canny edge detection method: 
		//1. Gaussian-blur 
		//2. Obtain gradient intensity and direction, 
		//3.Non-max suppression to determine is pixel better candidate than neighbors, 
		//4. Hysteresis thresholding to fine edge beg/end
		Imgproc.Canny(dstMat, dstMat, 3, 3 * 3, 3, false);
		
		//Fill dest-Img with zeroes
		Mat dest = new Mat();
		Core.add(dest, Scalar.all(0), dest);
		
		//Copy areas of image identified as edges(on black background)
		//This copy the pixels in the locations where they have non-zero values.
		srcMat.copyTo(dest, dstMat);
	}
	
	private void removeBackground(Mat srcMat) {
		srcMat.create(srcMat.size(), CvType.CV_8U);
		Mat dstMat = new Mat();
		Imgproc.cvtColor(srcMat, dstMat, Imgproc.COLOR_BGR2HSV);
		//Now let's split the three channels of the image:
		//Core.split(dstMat, hsvPlanes);	
	}
	
	private Mat removeBlkBorder(Mat image) throws Exception {
		//System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	    // reading image 
	    //Mat image = Highgui.imread(".\\testing2.jpg", Highgui.CV_LOAD_IMAGE_GRAYSCALE);
	    // clone the image 
	    Mat original = image.clone();
	    // thresholding the image to make a binary image
	    Imgproc.threshold(image, image, 100, 255, Imgproc.THRESH_BINARY_INV);
	    // find the center of the image
	    double[] centers = {(double)image.width()/2, (double)image.height()/2};
	    Point image_center = new Point(centers);

	    // finding the contours
	    ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
	    Mat hierarchy = new Mat();
	    Imgproc.findContours(image, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

	    // finding best bounding rectangle for a contour whose distance is closer to the image center that other ones
	    double d_min = Double.MAX_VALUE;
	    Rect rect_min = new Rect();
	    for (MatOfPoint contour : contours) {
	        Rect rec = Imgproc.boundingRect(contour);
	        // find the best candidates
	        if (rec.height > image.height()/2 & rec.width > image.width()/2)            
	            continue;
	        Point pt1 = new Point((double)rec.x, (double)rec.y);
	        Point center = new Point(rec.x+(double)(rec.width)/2, rec.y + (double)(rec.height)/2);
	        double d = Math.sqrt(Math.pow((double)(pt1.x-image_center.x),2) + Math.pow((double)(pt1.y -image_center.y), 2));            
	        if (d < d_min)
	        {
	            d_min = d;
	            rect_min = rec;
	        }                   
	    }
	    // slicing the image for result region
	    int pad = 5;        
	    rect_min.x = rect_min.x - pad;
	    rect_min.y = rect_min.y - pad;

	    rect_min.width = rect_min.width + 2*pad;
	    rect_min.height = rect_min.height + 2*pad;

	    Mat result = original.submat(rect_min);     
	    //Highgui.imwrite("result.png", result);
	    return result;
	}
	
	//1. Use ImageIO to read Image file...
	private BufferedImage readImage(String imgFPath) throws IOException {
		File fInputImg = new File(imgFPath);
		BufferedImage bufImg = null;
		
		//Detect mime-type of imgFPath...
		Tika tika = new Tika();
		String mimeType = tika.detect(fInputImg);
		if (mimeType != null && mimeType.contains("pdf")) {
			byte[] byteAryImg = this.generateImgFromPDF(imgFPath, "png");
			
			ByteArrayInputStream byteAryIS = new ByteArrayInputStream(byteAryImg);
			bufImg = ImageIO.read(byteAryIS);
		}
		else {
			//Not a PDF, create Image as-is...
			bufImg = ImageIO.read(fInputImg);
		}
		return bufImg;
	}
	
	//2. User Tess4J ImageHelper to Binarize BufferedImage to improve contrast ...i.e. turn image to black and white
	private BufferedImage imageToBinary(BufferedImage srcBufImg) throws IOException {
		BufferedImage binImg = ImageHelper.convertImageToBinary(srcBufImg);
		
		return binImg;
	}
	
	//3. Use OpenCV to convert BufferedImage to Mat
	private Mat bufferedImagetoMat(BufferedImage bufImage) throws IOException {
		ByteArrayOutputStream byteAryOS = new ByteArrayOutputStream();
		ImageIO.write(bufImage, "tif", byteAryOS);
		byteAryOS.flush();
		
		Mat dstMat = Imgcodecs.imdecode(new MatOfByte(byteAryOS.toByteArray()), Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
		return dstMat;
	}
	
	//3a New
	private Mat declareRectOfSkewArea(String imgFPath) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		ITesseract instance = new Tesseract();
		instance.setDatapath("/home/ubuntumac1/DevArea/Workspace/simRel-2018-09/demoTessJ4/tessdata");
		Mat srcMat = Imgcodecs.imread(imgFPath, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
		
		Mat tmpMat = srcMat.clone();
		Mat rectMat = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, tmpMat.size());
		Imgproc.erode(tmpMat, tmpMat, rectMat);
		
		//Create set of points before computing bounding box...
	    ArrayList<MatOfPoint> contourList = findContours(tmpMat, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
	    double maxArea = 0.0;
	    int maxAreaIdx = -1;
	    Collections.sort(contourList, new Comparator<MatOfPoint>() {
	        @Override
	        public int compare(MatOfPoint lhs, MatOfPoint rhs)
	        {
	        return Double.valueOf(Imgproc.contourArea(rhs)).compareTo(Imgproc.contourArea(lhs));
	        }
	    });

	    ArrayList<MatOfPoint> contourListMax = new ArrayList<>();
	    for (int idx = 0; idx < contourList.size(); idx++)
	    {
	        MatOfPoint contour = contourList.get(idx);

	        MatOfPoint2f c2f = new MatOfPoint2f(contour.toArray());
	        //MatOfPoint2f approx = new MatOfPoint2f();
	        //double epsilon = Imgproc.arcLength(c2f, true);
	        //Imgproc.approxPolyDP(c2f, approx, epsilon * 0.02, true);
	    }

	    
	    //Imgproc.minAreaRect(points);
		
		return tmpMat;
	}
	private ArrayList<MatOfPoint> findContours(Mat mat, int retrievalMode, int approximationMode)
	{
	    ArrayList<MatOfPoint> contourList = new ArrayList<>();
	    Mat hierarchy = new Mat();
	    Imgproc.findContours(mat, contourList, hierarchy, retrievalMode, approximationMode);
	    hierarchy.release();

	    return contourList;
	}

	
	//4. Enhance Mag Image using OpenCV fastNNlMeanDenoising and detailEnhance methods...
	private Mat denoiseAndDetainEnhanceImage(Mat inputMat) {
		Mat srcMat = inputMat.clone();
		Photo.fastNlMeansDenoising(inputMat, srcMat, 40, 10, 10);
		
		return srcMat;
	}
	
	//5. Convert Mat Image back to BufferedImage...
	private BufferedImage matToBufferedImage(Mat srcMat, String imgExt) throws Exception {
		MatOfByte matOfByte = new MatOfByte();
		Imgcodecs.imencode(imgExt, srcMat, matOfByte);
		byte[] byteAry = matOfByte.toArray();
		
		BufferedImage bufImg = ImageIO.read(new ByteArrayInputStream(byteAry));
		return bufImg;
	}
	
	//6. De-skew BufferedImage with minimum de-skew angle threshold of 0.05..
	private BufferedImage imageDeSkew(BufferedImage bufImg) throws IOException {
		double skewThreshold = 0.05;
		
		//Use Tess4J to de-skew image...
		ImageDeskew imgDeSkew = new ImageDeskew(bufImg);
		double skewAngle = imgDeSkew.getSkewAngle();
		System.out.println("Skew-angle: [" + skewAngle + "]");
		
		if (skewAngle > skewThreshold || skewAngle > -skewThreshold) {
			bufImg = ImageUtil.rotate(bufImg, -skewAngle, bufImg.getWidth()/2, bufImg.getHeight()/2);
		}
		
		return bufImg;
	}
	
	//6. De-skew BufferedImage
	private Mat imageDeSkew(Mat src, double angle) {
		Point center = new Point(src.width()/2, src.height()/2);
		Mat rotatdImg = Imgproc.getRotationMatrix2D(center, angle, 1.0);	//1.0 implies 100 % scale!
		
		Size size = new Size(src.width(), src.height());
		Imgproc.warpAffine(src, src, rotatdImg, size, Imgproc.INTER_LINEAR + Imgproc.CV_WARP_FILL_OUTLIERS);
		return src;
	}
	
	//7. Convert BufferedImage to byte[] for caller to consume...
	private byte[] imageWriter(String format, BufferedImage bufImg) throws IOException {
		RenderedImage rendImg = (RenderedImage) bufImg;
		ByteArrayOutputStream byteAryOS = new ByteArrayOutputStream();
		
		ImageWriter writer = ImageIO.getImageWritersByFormatName(format).next();
		ImageOutputStream imgOS = ImageIO.createImageOutputStream(byteAryOS);
		writer.setOutput(imgOS);
		writer.write(rendImg);
		
		byte[] byteAry = byteAryOS.toByteArray();
		byteAryOS.flush();
		byteAryOS.close();
		
		return byteAry;
	}
	
	private void computeSkew(String inFName) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		ITesseract instance = new Tesseract();
		instance.setDatapath("/home/ubuntumac1/DevArea/Workspace/simRel-2018-09/demoTessJ4/tessdata");
		//Load image in Grey-scale...
		Mat img = Imgcodecs.imread(inFName, Imgcodecs.IMREAD_GRAYSCALE);
		
		//Binarize image...
		Imgproc.threshold(img, img, 200, 255, Imgproc.THRESH_BINARY);
		
		//Invert colors i.e. White pixels represent Objects, and Black pixels  represent background
		Core.bitwise_not(img, img);
		Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
		
		//Declare rect-shaped structuring element and call erode function...
		Imgproc.erode(img, img, element);
		
		//Find all White pixels...
		Mat wLocMat  = Mat.zeros(img.size(), img.type());
		Core.findNonZero(img, wLocMat);
		
		//Create an empty Mat for the function..
		MatOfPoint matOfPt = new MatOfPoint();
		
		//Translate MatOfPoint to MatOfPoint2f for the next step...
		MatOfPoint2f matOfPt2f = new MatOfPoint2f();
		matOfPt2f.convertTo(matOfPt2f, CvType.CV_32FC2);
		
		//Get rotated rect of the White pixels...
		RotatedRect rotatedRect = Imgproc.minAreaRect(matOfPt2f);
		
		Point[] vertices = new Point[4];
		rotatedRect.points(vertices);
		
		List<MatOfPoint> boxContours = new ArrayList<>();
		boxContours.add(new MatOfPoint(vertices));
		Imgproc.drawContours(img, boxContours, 0, new Scalar(128, 128, 128), -1);
		
		double resultAngle = rotatedRect.angle;
		if (rotatedRect.size.width > rotatedRect.size.height) {
			rotatedRect.angle += 90.f;
		}
		else {
			rotatedRect.angle = rotatedRect.angle < -45 ? rotatedRect.angle + 90.0f : rotatedRect.angle;
		}
		
		Mat result = deskew(Imgcodecs.imread(inFName), rotatedRect.angle);
		Imgcodecs.imwrite("/DevArea/Workspace/tess4joutput", result);
	}
	
	private Mat deskew(Mat srcMat, double angle) {
		Point center = new Point(srcMat.width()/2, srcMat.height()/2);
		Mat rotatdImg = Imgproc.getRotationMatrix2D(center, angle, 1.0);
		
		Size size = new Size(srcMat.width(), srcMat.height());
		Imgproc.warpAffine(srcMat, srcMat, rotatdImg, size, Imgproc.INTER_LINEAR + Imgproc.CV_WARP_FILL_OUTLIERS);
		
		return srcMat;
	}
	
	private void boundingBox (String inFile) throws Exception {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		ITesseract instance = new Tesseract();
		instance.setDatapath("/home/ubuntumac1/DevArea/Workspace/simRel-2018-09/demoTessJ4/tessdata");
		
		String inFPathNoExt = inFile.substring(0, inFile.lastIndexOf("."));
		String inFName = inFPathNoExt.substring(inFPathNoExt.lastIndexOf("/") + 1);
		String outFile = inFile.substring(0, inFile.lastIndexOf("/") + 1) + "output/" + inFName;
		String outputImgFPath = outFile + ".png";
		
		
		//Load image in Grey-scale...
		Mat img = Imgcodecs.imread(inFile, Imgcodecs.IMREAD_GRAYSCALE);
		Core.bitwise_not(img, img);
		
		BufferedImage bufImg = this.matToBufferedImage(img, ".png");
		//Use Tess4J to de-skew image...
		ImageDeskew imgDeSkew = new ImageDeskew(bufImg);
		double skewAngle = imgDeSkew.getSkewAngle();
		System.out.println("Skew-angle: [" + skewAngle + "]");
		
		img = this.imageDeSkew(img, skewAngle);
		
		//Get Bounding  box
		RotatedRect rect = this.getRotatedRect(img);
		
		//Get Rotation
		//Mat rotMat = Imgproc.getRotationMatrix2D(rect.center, skewAngle, 1);
		Mat rotMat = this.imageDeSkew(img, skewAngle);
		
		//Crop Image
		Size boxSize = rect.size;
		if (rect.angle < -45.) {
			double width = boxSize.width;
			double height = boxSize.height;
			boxSize.width = height;
			boxSize.height = width;
		}
		Mat cropped = new Mat();
		Imgproc.getRectSubPix(img, boxSize, rect.center, cropped);
		
		Imgcodecs.imwrite(outputImgFPath, cropped);
		
		//Convert created image to PDF...
		List<RenderedFormat> lstRendFmt = Arrays.asList(RenderedFormat.PDF);
		File pngFPath = new File(outputImgFPath);
		instance.createDocuments(pngFPath.getAbsolutePath(), outFile, lstRendFmt);		
		
	}
//	private void cropImage(Mat img) {
//		Size boxSize = img.size();
//		if (img.)
//	}
	private void findContours1(Mat deskewedImg) {
		List<MatOfPoint> contours = new ArrayList<>();
		Mat dest = Mat.zeros(deskewedImg.size(), CvType.CV_8UC3);
		Scalar white = new Scalar(255, 255, 255);

		// Find contours
		Imgproc.findContours(deskewedImg, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

		// Draw contours in dest Mat
		Imgproc.drawContours(dest, contours, -1, white);
		
		//Find best fit rect for each contour
		Scalar green = new Scalar(81, 190, 0);
		RotatedRect rotatedRect = null;
		for (MatOfPoint contour: contours) {
		    rotatedRect = Imgproc.minAreaRect(new MatOfPoint2f(contour.toArray()));
		    //drawRotatedRect(dest, rotatedRect, green, 4);
		}
		
	}
	public static void drawRotatedRect(Mat image, RotatedRect rotatedRect, Scalar color, int thickness) {
	    Point[] vertices = new Point[4];
	    rotatedRect.points(vertices);
	    MatOfPoint points = new MatOfPoint(vertices);
	    Imgproc.drawContours(image, Arrays.asList(points), -1, color, thickness);
	}
	//This is bounding box! Not rotated at this time
	public RotatedRect getRotatedRect(Mat binImg) {
		RotatedRect rect = null;

		Mat points = Mat.zeros(binImg.size(),binImg.type());
		Core.findNonZero(binImg, points);

		MatOfPoint mpoints = new MatOfPoint(points);
		MatOfPoint2f points2f = new MatOfPoint2f(mpoints.toArray());

		if (points2f.rows() > 0) {
		    rect = Imgproc.minAreaRect(points2f);
		}	
		return rect;
	}
	/*
	void demo(String inFName) {
		//Get min bounding box for the image
		Mat srcMat = Imgcodecs.imread(inFName, Imgcodecs.IMREAD_GRAYSCALE);
		srcMat = 
	}
	
	void test() {
		import cv2
		import numpy as np

		# get the minimum bounding box for the chip image
		image = cv2.imread("./chip1.png", cv2.IMREAD_COLOR)
		image = image[10:-10,10:-10]
		imgray = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[...,0]
		ret, thresh = cv2.threshold(imgray, 20, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
		mask = 255 - thresh
		_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

		maxArea = 0
		best = None
		for contour in contours:
		  area = cv2.contourArea(contour)
		  if area > maxArea :
		    maxArea = area
		    best = contour

		rect = cv2.minAreaRect(best)
		box = cv2.boxPoints(rect)
		box = np.int0(box)

		#crop image inside bounding box
		scale = 1  # cropping margin, 1 == no margin
		W = rect[1][0]
		H = rect[1][1]

		Xs = [i[0] for i in box]
		Ys = [i[1] for i in box]
		x1 = min(Xs)
		x2 = max(Xs)
		y1 = min(Ys)
		y2 = max(Ys)

		angle = rect[2]
		rotated = False
		if angle < -45:
		    angle += 90
		    rotated = True

		center = (int((x1+x2)/2), int((y1+y2)/2))
		size = (int(scale*(x2-x1)), int(scale*(y2-y1)))

		M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

		cropped = cv2.getRectSubPix(image, size, center)
		cropped = cv2.warpAffine(cropped, M, size)
	}
*/
}

////////////////////////

package com.example.demo;

import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.LongAdder;

import javax.imageio.ImageIO;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageOutputStream;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.apache.tika.Tika;
import org.bytedeco.javacpp.indexer.UByteBufferIndexer;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.photo.Photo;

import com.recognition.software.jdeskew.ImageDeskew;
import com.recognition.software.jdeskew.ImageUtil;

import net.sourceforge.tess4j.ITesseract;
import net.sourceforge.tess4j.ITesseract.RenderedFormat;
import net.sourceforge.tess4j.Tesseract;
import net.sourceforge.tess4j.util.ImageHelper;

public class DemoTess4J {
	
	//Location of opencv binaries for Macos: https://github.com/opencv/opencv/archive/3.4.2.tar.gz
	
	//Note!!! Tess4J requires the ff to set for Tesseract to work!!!
	//-Djna.library.path=<Location of platform specific native dll/so>
	//-Dsun.java2d.cmm=sun.java2d.cmm.kcms.KcmsServiceProvider =>Improves Java 8/9/etc performance!!!!

	private static final int RUNS=10;
	public static void main(String[] args) {
		
//		Long timeNow = System.currentTimeMillis();
//		File[] files = new File("/home/ubuntumac1/DevArea/Workspace/simRel-2018-09/demoTessJ4/test-data/").listFiles();
//		System.out.println(String.format("Available procs: %d",  Runtime.getRuntime().availableProcessors()));
//		System.out.println(String.format("There are %d png files to OCR in paralled.",  files.length));
//		
//		final LongAdder totalErrs = new LongAdder();
//		
//		DemoTess4J demoP = new DemoTess4J();
//		for (int runIdx=1; runIdx<RUNS; runIdx++) {
//			final LongAdder errInCurRun = new LongAdder();
//			
//			Arrays.stream(files).parallel().forEach((file) -> {
//				try {
//					demoP.renderDocument(file.getAbsolutePath().replace("\\", "/"));
//				}
//				catch (Exception ex) {
//					errInCurRun.increment();
//				}
//			});
//			System.out.println(String.format("\tRun %d -> Errors: %d/%d",  runIdx, errInCurRun.intValue(), files.length));
//			totalErrs.add(errInCurRun.intValue());
//		}
//		int cntErrs = totalErrs.intValue();
//		System.out.println(String.format("Total Count-Errs: %d / Err Percentage: %.2f%%", 
//				cntErrs, cntErrs / (double) files.length *100 / RUNS));
//		
//		Long timeEnd = System.currentTimeMillis();
//		System.out.println("Elapsed Time [" + (timeEnd - timeNow) + "]");
		
		String inFile = "/home/ubuntumac1/DevArea/Workspace/simRel-2018-09/demoTessJ4/test-data/eurotext_deskew.png";
		DemoTess4J demo = new DemoTess4J();
		try {
			//demo.renderDocument(inFile);
			demo.declareRectOfSkewArea(inFile);
		} 
		catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private void renderDocument(String inFile) throws Exception {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		ITesseract instance = new Tesseract();
		instance.setDatapath("/home/ubuntumac1/DevArea/Workspace/simRel-2018-09/demoTessJ4/tessdata");
		
		String inFPathNoExt = inFile.substring(0, inFile.lastIndexOf("."));
		String inFName = inFPathNoExt.substring(inFPathNoExt.lastIndexOf("/") + 1);
		String outFile = inFile.substring(0, inFile.lastIndexOf("/") + 1) + "output/" + inFName;
		String outputImgFPath = outFile + ".png";
		
		try {
			BufferedImage srcBufImg = readImage(inFile);
			srcBufImg = this.imageToBinary(srcBufImg);
			Mat bufImgToMat = this.bufferedImagetoMat(srcBufImg);
			//bufImgToMat = this.removeBlkBorder(bufImgToMat);
			Mat enhancedImgMat = this.denoiseAndDetainEnhanceImage(bufImgToMat);
			BufferedImage mat2BufImage = this.matToBufferedImage(enhancedImgMat, ".png");
			BufferedImage deskewedImg = this.imageDeSkew(mat2BufImage);
			byte[] deskewedByteAryImg = this.imageWriter("png", deskewedImg);
			
			Mat result = Imgcodecs.imdecode(new MatOfByte(deskewedByteAryImg), Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
			//result = this.removeBlkBorder(result);
			//result = this.makeBlackTransparent(result);
			Imgcodecs.imwrite(outputImgFPath, result);
			
			//Convert created image to PDF...
			List<RenderedFormat> lstRendFmt = Arrays.asList(RenderedFormat.PDF);
			File pngFPath = new File(outputImgFPath);
			instance.createDocuments(pngFPath.getAbsolutePath(), outFile, lstRendFmt);		
		}
		catch(Exception ex) {
			ex.printStackTrace();
		}
	}
	
	private byte[] generateImgFromPDF(String inputPdfFPath, String format) throws IOException {
		ByteArrayOutputStream byteAryOS = null;
		
		PDDocument doc = null;
		try {
			doc = PDDocument.load(new File(inputPdfFPath));
			PDFRenderer pdfRendr = new PDFRenderer(doc);
			
			byteAryOS = new ByteArrayOutputStream();
			int pageCnt = doc.getNumberOfPages();
			
			byteAryOS = new ByteArrayOutputStream();
			ImageWriter writer = ImageIO.getImageWritersByFormatName(format).next();
			ImageOutputStream imgOS = ImageIO.createImageOutputStream(byteAryOS);

			for (int page=0; page < pageCnt; page++) {
				BufferedImage bufImg = pdfRendr.renderImageWithDPI(page, 300, ImageType.BINARY);
				
				RenderedImage rendImg = (RenderedImage) bufImg;
				byteAryOS = new ByteArrayOutputStream();
				//ImageWriter writer = ImageIO.getImageWritersByFormatName(format).next();
				//ImageOutputStream imgOS = ImageIO.createImageOutputStream(byteAryOS);
				
				writer.setOutput(imgOS);
				writer.write(rendImg);
			}
		}
		finally {
			doc.close();
			byteAryOS.flush();
			byteAryOS.close();
		}
		
		return byteAryOS.toByteArray();
	}
	
	/**

	 * Make the black background of a PNG-Bitmap-Image transparent.
	 * code based on example at http://j.mp/1uCxOV5
	 * @Param image png bitmap image
	 * @return output image
	 */

	private Mat makeBlackTransparent(Mat image) {
	    // convert image to matrix
	    Mat src = new Mat(image.width(), image.height(), CvType.CV_8UC4);

	    // init new matrices
	    Mat dst = new Mat(image.width(), image.height(), CvType.CV_8UC4);
	    Mat tmp = new Mat(image.width(), image.height(), CvType.CV_8UC4);
	    Mat alpha = new Mat(image.width(), image.height(), CvType.CV_8UC4);

	    // convert image to grayscale
	    Imgproc.cvtColor(src, tmp, Imgproc.COLOR_BGR2GRAY);

	    // threshold the image to create alpha channel with complete transparency in black background region and zero transparency in foreground object region.
	    Imgproc.threshold(tmp, alpha, 100, 255, Imgproc.THRESH_BINARY);

	    // split the original image into three single channel.
	    List<Mat> rgb = new ArrayList<Mat>(3);
	    Core.split(src, rgb);

	    // Create the final result by merging three single channel and alpha(BGRA order)
	    List<Mat> rgba = new ArrayList<Mat>(4);
	    rgba.add(rgb.get(0));
	    rgba.add(rgb.get(1));
	    rgba.add(rgb.get(2));
	    rgba.add(alpha);
	    Core.merge(rgba, dst);

	    // convert matrix to output bitmap
	    //Mat output = Bitmap.createBitmap(image.width(), image.height(), Bitmap.Config.ARGB_8888);
	    Mat output = new Mat(image.width(), image.height(), Imgproc.CV_RGBA2mRGBA);
		Imgproc.cvtColor(image, output, Imgproc.CV_RGBA2mRGBA);
	    return output;
	}
	
	private void edgeDetector(Mat srcMat) {
		Mat dstMat = new Mat();
		
		//Convert image to grey
		Imgproc.cvtColor(srcMat, dstMat, Imgproc.COLOR_BGR2GRAY);
		
		//Mat detectedEdges = new Mat();
		Imgproc.blur(dstMat, dstMat, new Size(3, 3));
		
		//Apply Canny edge detection method: 
		//1. Gaussian-blur 
		//2. Obtain gradient intensity and direction, 
		//3.Non-max suppression to determine is pixel better candidate than neighbors, 
		//4. Hysteresis thresholding to fine edge beg/end
		Imgproc.Canny(dstMat, dstMat, 3, 3 * 3, 3, false);
		
		//Fill dest-Img with zeroes
		Mat dest = new Mat();
		Core.add(dest, Scalar.all(0), dest);
		
		//Copy areas of image identified as edges(on black background)
		//This copy the pixels in the locations where they have non-zero values.
		srcMat.copyTo(dest, dstMat);
	}
	
	private void removeBackground(Mat srcMat) {
		srcMat.create(srcMat.size(), CvType.CV_8U);
		Mat dstMat = new Mat();
		Imgproc.cvtColor(srcMat, dstMat, Imgproc.COLOR_BGR2HSV);
		//Now let's split the three channels of the image:
		//Core.split(dstMat, hsvPlanes);	
	}
	
	private Mat removeBlkBorder(Mat image) throws Exception {
		//System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	    // reading image 
	    //Mat image = Highgui.imread(".\\testing2.jpg", Highgui.CV_LOAD_IMAGE_GRAYSCALE);
	    // clone the image 
	    Mat original = image.clone();
	    // thresholding the image to make a binary image
	    Imgproc.threshold(image, image, 100, 255, Imgproc.THRESH_BINARY_INV);
	    // find the center of the image
	    double[] centers = {(double)image.width()/2, (double)image.height()/2};
	    Point image_center = new Point(centers);

	    // finding the contours
	    ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
	    Mat hierarchy = new Mat();
	    Imgproc.findContours(image, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

	    // finding best bounding rectangle for a contour whose distance is closer to the image center that other ones
	    double d_min = Double.MAX_VALUE;
	    Rect rect_min = new Rect();
	    for (MatOfPoint contour : contours) {
	        Rect rec = Imgproc.boundingRect(contour);
	        // find the best candidates
	        if (rec.height > image.height()/2 & rec.width > image.width()/2)            
	            continue;
	        Point pt1 = new Point((double)rec.x, (double)rec.y);
	        Point center = new Point(rec.x+(double)(rec.width)/2, rec.y + (double)(rec.height)/2);
	        double d = Math.sqrt(Math.pow((double)(pt1.x-image_center.x),2) + Math.pow((double)(pt1.y -image_center.y), 2));            
	        if (d < d_min)
	        {
	            d_min = d;
	            rect_min = rec;
	        }                   
	    }
	    // slicing the image for result region
	    int pad = 5;        
	    rect_min.x = rect_min.x - pad;
	    rect_min.y = rect_min.y - pad;

	    rect_min.width = rect_min.width + 2*pad;
	    rect_min.height = rect_min.height + 2*pad;

	    Mat result = original.submat(rect_min);     
	    //Highgui.imwrite("result.png", result);
	    return result;
	}
	
	//1. Use ImageIO to read Image file...
	private BufferedImage readImage(String imgFPath) throws IOException {
		File fInputImg = new File(imgFPath);
		BufferedImage bufImg = null;
		
		//Detect mime-type of imgFPath...
		Tika tika = new Tika();
		String mimeType = tika.detect(fInputImg);
		if (mimeType != null && mimeType.contains("pdf")) {
			byte[] byteAryImg = this.generateImgFromPDF(imgFPath, "png");
			
			ByteArrayInputStream byteAryIS = new ByteArrayInputStream(byteAryImg);
			bufImg = ImageIO.read(byteAryIS);
		}
		else {
			//Not a PDF, create Image as-is...
			bufImg = ImageIO.read(fInputImg);
		}
		return bufImg;
	}
	
	//2. User Tess4J ImageHelper to Binarize BufferedImage to improve contrast ...i.e. turn image to black and white
	private BufferedImage imageToBinary(BufferedImage srcBufImg) throws IOException {
		BufferedImage binImg = ImageHelper.convertImageToBinary(srcBufImg);
		
		return binImg;
	}
	
	//3. Use OpenCV to convert BufferedImage to Mat
	private Mat bufferedImagetoMat(BufferedImage bufImage) throws IOException {
		ByteArrayOutputStream byteAryOS = new ByteArrayOutputStream();
		ImageIO.write(bufImage, "tif", byteAryOS);
		byteAryOS.flush();
		
		Mat dstMat = Imgcodecs.imdecode(new MatOfByte(byteAryOS.toByteArray()), Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
		return dstMat;
	}
	
	//3a New
	private Mat declareRectOfSkewArea(String imgFPath) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		ITesseract instance = new Tesseract();
		instance.setDatapath("/home/ubuntumac1/DevArea/Workspace/simRel-2018-09/demoTessJ4/tessdata");
		Mat srcMat = Imgcodecs.imread(imgFPath, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
		
		Mat tmpMat = srcMat.clone();
		Mat rectMat = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, tmpMat.size());
		Imgproc.erode(tmpMat, tmpMat, rectMat);
		
		//Create set of points before computing bounding box...
	    ArrayList<MatOfPoint> contourList = findContours(tmpMat, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
	    double maxArea = 0.0;
	    int maxAreaIdx = -1;
	    Collections.sort(contourList, new Comparator<MatOfPoint>() {
	        @Override
	        public int compare(MatOfPoint lhs, MatOfPoint rhs)
	        {
	        return Double.valueOf(Imgproc.contourArea(rhs)).compareTo(Imgproc.contourArea(lhs));
	        }
	    });

	    ArrayList<MatOfPoint> contourListMax = new ArrayList<>();
	    for (int idx = 0; idx < contourList.size(); idx++)
	    {
	        MatOfPoint contour = contourList.get(idx);

	        MatOfPoint2f c2f = new MatOfPoint2f(contour.toArray());
	        //MatOfPoint2f approx = new MatOfPoint2f();
	        //double epsilon = Imgproc.arcLength(c2f, true);
	        //Imgproc.approxPolyDP(c2f, approx, epsilon * 0.02, true);
	    }

	    
	    //Imgproc.minAreaRect(points);
		
		return tmpMat;
	}
	private ArrayList<MatOfPoint> findContours(Mat mat, int retrievalMode, int approximationMode)
	{
	    ArrayList<MatOfPoint> contourList = new ArrayList<>();
	    Mat hierarchy = new Mat();
	    Imgproc.findContours(mat, contourList, hierarchy, retrievalMode, approximationMode);
	    hierarchy.release();

	    return contourList;
	}

	
	//4. Enhance Mag Image using OpenCV fastNNlMeanDenoising and detailEnhance methods...
	private Mat denoiseAndDetainEnhanceImage(Mat inputMat) {
		Mat srcMat = inputMat.clone();
		Photo.fastNlMeansDenoising(inputMat, srcMat, 40, 10, 10);
		
		return srcMat;
	}
	
	//5. Convert Mat Image back to BufferedImage...
	private BufferedImage matToBufferedImage(Mat srcMat, String imgExt) throws Exception {
		MatOfByte matOfByte = new MatOfByte();
		Imgcodecs.imencode(imgExt, srcMat, matOfByte);
		byte[] byteAry = matOfByte.toArray();
		
		BufferedImage bufImg = ImageIO.read(new ByteArrayInputStream(byteAry));
		return bufImg;
	}
	
	//6. De-skew BufferedImage with minimum de-skew angle threshold of 0.05..
	private BufferedImage imageDeSkew(BufferedImage bufImg) throws IOException {
		double skewThreshold = 0.05;
		
		//Use Tess4J to de-skew image...
		ImageDeskew imgDeSkew = new ImageDeskew(bufImg);
		double skewAngle = imgDeSkew.getSkewAngle();
		
		if (skewAngle > skewThreshold || skewAngle > -skewThreshold) {
			bufImg = ImageUtil.rotate(bufImg, -skewAngle, bufImg.getWidth()/2, bufImg.getHeight()/2);
		}
		
		return bufImg;
	}
	
	//6. De-skew BufferedImage
	private Mat imageDeSkew(Mat src, double angle) {
		Point center = new Point(src.width()/2, src.height()/2);
		Mat rotatdImg = Imgproc.getRotationMatrix2D(center, angle, 1.0);	//1.0 implies 100 % scale!
		
		Size size = new Size(src.width(), src.height());
		Imgproc.warpAffine(src, src, rotatdImg, size, Imgproc.INTER_LINEAR + Imgproc.CV_WARP_FILL_OUTLIERS);
		return src;
	}
	
	//7. Convert BufferedImage to byte[] for caller to consume...
	private byte[] imageWriter(String format, BufferedImage bufImg) throws IOException {
		RenderedImage rendImg = (RenderedImage) bufImg;
		ByteArrayOutputStream byteAryOS = new ByteArrayOutputStream();
		
		ImageWriter writer = ImageIO.getImageWritersByFormatName(format).next();
		ImageOutputStream imgOS = ImageIO.createImageOutputStream(byteAryOS);
		writer.setOutput(imgOS);
		writer.write(rendImg);
		
		byte[] byteAry = byteAryOS.toByteArray();
		byteAryOS.flush();
		byteAryOS.close();
		
		return byteAry;
	}
	
	private void computeSkew(String inFName) {
		//Load image in Grey-scale...
		Mat img = Imgcodecs.imread(inFName, Imgcodecs.IMREAD_GRAYSCALE);
		
		//Binarize image...
		Imgproc.threshold(img, img, 200, 255, Imgproc.THRESH_BINARY);
		
		//Invert colors i.e. White pixels represent Objects, and Black pixels  represent background
		Core.bitwise_not(img, img);
		Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
		
		//Declare rect-shaped structuring element and call erode function...
		Imgproc.erode(img, img, element);
		
		//Find all White pixels...
		Mat wLocMat  = Mat.zeros(img.size(), img.type());
		Core.findNonZero(img, wLocMat);
		
		//Create an empty Mat for the function..
		MatOfPoint matOfPt = new MatOfPoint();
		
		//Translate MatOfPoint to MatOfPoint2f for the next step...
		MatOfPoint2f matOfPt2f = new MatOfPoint2f();
		matOfPt2f.convertTo(matOfPt2f, CvType.CV_32FC2);
		
		//Get rotated rect of the White pixels...
		RotatedRect rotatedRect = Imgproc.minAreaRect(matOfPt2f);
		
		Point[] vertices = new Point[4];
		rotatedRect.points(vertices);
		
		List<MatOfPoint> boxContours = new ArrayList<>();
		boxContours.add(new MatOfPoint(vertices));
		Imgproc.drawContours(img, boxContours, 0, new Scalar(128, 128, 128), -1);
		
		double resultAngle = rotatedRect.angle;
		if (rotatedRect.size.width > rotatedRect.size.height) {
			rotatedRect.angle += 90.f;
		}
		else {
			rotatedRect.angle = rotatedRect.angle < -45 ? rotatedRect.angle + 90.0f : rotatedRect.angle;
		}
		
		Mat result = deskew(Imgcodecs.imread(inFName), rotatedRect.angle);
		Imgcodecs.imwrite("/DevArea/Workspace/tess4joutput", result);
	}
	
	private Mat deskew(Mat srcMat, double angle) {
		Point center = new Point(srcMat.width()/2, srcMat.height()/2);
		Mat rotatdImg = Imgproc.getRotationMatrix2D(center, angle, 1.0);
		
		Size size = new Size(srcMat.width(), srcMat.height());
		Imgproc.warpAffine(srcMat, srcMat, rotatdImg, size, Imgproc.INTER_LINEAR + Imgproc.CV_WARP_FILL_OUTLIERS);
		
		return srcMat;
	}

	/*
	void demo(String inFName) {
		//Get min bounding box for the image
		Mat srcMat = Imgcodecs.imread(inFName, Imgcodecs.IMREAD_GRAYSCALE);
		srcMat = 
	}
	
	void test() {
		import cv2
		import numpy as np

		# get the minimum bounding box for the chip image
		image = cv2.imread("./chip1.png", cv2.IMREAD_COLOR)
		image = image[10:-10,10:-10]
		imgray = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[...,0]
		ret, thresh = cv2.threshold(imgray, 20, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
		mask = 255 - thresh
		_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

		maxArea = 0
		best = None
		for contour in contours:
		  area = cv2.contourArea(contour)
		  if area > maxArea :
		    maxArea = area
		    best = contour

		rect = cv2.minAreaRect(best)
		box = cv2.boxPoints(rect)
		box = np.int0(box)

		#crop image inside bounding box
		scale = 1  # cropping margin, 1 == no margin
		W = rect[1][0]
		H = rect[1][1]

		Xs = [i[0] for i in box]
		Ys = [i[1] for i in box]
		x1 = min(Xs)
		x2 = max(Xs)
		y1 = min(Ys)
		y2 = max(Ys)

		angle = rect[2]
		rotated = False
		if angle < -45:
		    angle += 90
		    rotated = True

		center = (int((x1+x2)/2), int((y1+y2)/2))
		size = (int(scale*(x2-x1)), int(scale*(y2-y1)))

		M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

		cropped = cv2.getRectSubPix(image, size, center)
		cropped = cv2.warpAffine(cropped, M, size)
	}
*/
}
/*
Here is the algorithm to calculate the largest rectangle in a rotated image.
?
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
/**
  * Return a largest Rectangle that will fit in a rotated image
  * @param imgWidth Width of image
  * @param imgHeight Height of Image
  * @param rotAngDeg Rotation angle in degrees
  * @param type 0 = Largest Area, 1 = Smallest Area, 2 = Widest, 3 = Tallest
  * @return
  */
 public static Rectangle getLargestRectangle(double imageWidth, double imageHeight, double rotAngDeg, int type) {
  Rectangle rect = null;
   
  double rotateAngleDeg = rotAngDeg % 180d;
  if (rotateAngleDeg < 0d) {
   rotateAngleDeg += 360d;
   rotateAngleDeg = rotateAngleDeg % 180d;
  }
  double imgWidth = imageWidth;
  double imgHeight = imageHeight;
   
  if (rotateAngleDeg == 0 || rotateAngleDeg == 180) {
   // Angle is 0. No change needed
   rect = new Rectangle(0,0,(int)imgWidth,(int)imgHeight);
   return rect;
  }
   
  if (rotateAngleDeg == 90) {
   // Angle is 90. Width and height swapped
   rect = new Rectangle(0,0,(int)imgHeight,(int)imgWidth);
   return rect;
  }
   
  if (rotateAngleDeg > 90) {
   // Angle > 90 therefore angle = 90 - ("+rotateAngleDeg+" - 90) = "+(90 - (rotateAngleDeg - 90))
   rotateAngleDeg = 90 - (rotateAngleDeg - 90);
  }
   
  double rotateAngle = Math.toRadians(rotateAngleDeg);
  double sinRotAng = Math.sin(rotateAngle);
  double cosRotAng = Math.cos(rotateAngle);
  double tanRotAng = Math.tan(rotateAngle);
  // Point 1 of rotated rectangle
  double x1 = sinRotAng * imgHeight;
  double y1 = 0;
  // Point 2 of rotated rectangle
  double x2 = cosRotAng * imgWidth + x1;
  double y2 = sinRotAng * imgWidth;
  // Point 3 of rotated rectangle
  double x3 = x2 - x1;
  double y3 = y2 + cosRotAng * imgHeight;
  // Point 4 of rotated rectangle
  double x4 = 0;
  double y4 = y3 - y2;
  // MidPoint of rotated image
  double midx = x2 / 2;
  double midy = y3 / 2;
   
  // Angle for new rectangle (based on image width and height)
  double imgAngle = Math.atan(imgHeight / imgWidth);
  double imgRotAngle = Math.atan(imgWidth / imgHeight);
  double tanImgAng = Math.tan(imgAngle);
  double tanImgRotAng = Math.tan(imgRotAngle);
  // X Point for new rectangle on bottom line
  double ibx1 = midy / tanImgAng + midx;
  double ibx2 = midy * tanImgAng + midx;
   
  // First intersecting lines
  // y = ax + b  ,  y = cx + d  ==>  x = (d - b) / (a - c)
  double a = y2 / x3;
  double b = tanRotAng * -x1;
  double c = -imgHeight / imgWidth;
  double d = tanImgAng * ibx1;
   
  // Intersecting point 1
  double ix1 = (d - b) / (a - c);
  double iy1 = a * ix1 + b;
   
  // Second intersecting lines
  c = -imgWidth / imgHeight;
  d = tanImgRotAng * ibx2;
   
  // Intersecting point 2
  double ix2 = (d - b) / (a - c);
  double iy2 = a * ix2 + b;
   
  // Work out smallest rectangle
  double radx1 = Math.abs(midx - ix1);
  double rady1 = Math.abs(midy - iy1);
  double radx2 = Math.abs(midx - ix2);
  double rady2 = Math.abs(midy - iy2);
  // Work out area of rectangles
  double area1 = radx1 * rady1;
  double area2 = radx2 * rady2;
  // Rectangle (x,y,width,height)
  Rectangle rect1 = new Rectangle((int)Math.round(midx-radx1),(int)Math.round(midy-rady1),
    (int)Math.round(radx1*2),(int)Math.round(rady1*2));
   
  // Rectangle (x,y,width,height)
  Rectangle rect2 = new Rectangle((int)Math.round(midx-radx2),(int)Math.round(midy-rady2),
    (int)Math.round(radx2*2),(int)Math.round(rady2*2));
   
  switch (type) {
   case 0: rect = (area1 > area2 ? rect1 : rect2); break;
   case 1: rect = (area1 < area2 ? rect1 : rect2); break;
   case 2: rect = (radx1 > radx2 ? rect1 : rect2); break;
   case 3: rect = (rady1 > rady2 ? rect1 : rect2); break;
  }
   
  return rect;
 }

Here is some code you want want to use for your own testing. It uses JAI for rotating and cropping but really you can use any library for this.

?
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
public static void main(String[] args) throws IOException {
 
 BufferedImage src = ImageIO.read(new File("MyTestImage.jpg"));
 String outType = "jpg";
 int imgWidth = 1280;
 int imgHeight = 851;
 
 double rotateAngle = 20d;
 BufferedImage rotImg = rotateImage(src, rotateAngle);
 ImageIO.write(rotImg, outType, new File("RotateTest."+outType));
      
 Rectangle rectLargest = getLargestRectangle(imgWidth,imgHeight,rotateAngle,0);
 Rectangle rectTallest = getLargestRectangle(imgWidth,imgHeight,rotateAngle,3);
      
 BufferedImage rotCropImage = cropImage(rotImg, (int)rectLargest.getX(), (int)rectLargest.getY(), (int)rectLargest.getWidth(), (int)rectLargest.getHeight());
 ImageIO.write(rotCropImage, outType, new File("RotateCropTest_LG."+outType));
 
 rotCropImage = cropImage(rotImg, (int)rectTallest.getX(), (int)rectTallest.getY(), (int)rectTallest.getWidth(), (int)rectTallest.getHeight());
 ImageIO.write(rotCropImage, outType, new File("RotateCropTest_TA."+outType));
}
     
public static BufferedImage rotateImage(BufferedImage image, double angle)
{
 // Gets the angle (converting it to radians).
 float rangle = (float)Math.toRadians(angle);
 // Gets the rotation center.
 float centerX = 0f; float centerY = 0f;
 centerX = image.getWidth()/2f;
 centerY = image.getHeight()/2f;
 
 // Rotates the original image.
 ParameterBlock pb = new ParameterBlock();
 pb.addSource(image);
 pb.add(centerX);
 pb.add(centerY);
 pb.add(rangle);
 pb.add(new InterpolationBilinear());
 // Creates a new, rotated image and uses it on the DisplayJAI component
 RenderedOp result = JAI.create("rotate", pb);
 return result.getAsBufferedImage();
}
 
public static BufferedImage cropImage(BufferedImage image, int topLeftX, int topLeftY, int width, int height)
{
 // Crops the original image.
 ParameterBlock pb = new ParameterBlock();
 pb.addSource(image);
 pb.add((float)topLeftX);
 pb.add((float)topLeftY);
 pb.add((float)width);
 pb.add((float)height);
 // Creates a new, rotated image and uses it on the DisplayJAI component
 RenderedOp result = JAI.create("crop", pb);
 return result.getAsBufferedImage();
}

////Another example
package com.example.demo;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class DemoTess4J1 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}
	
	public void scanDocument(Bitmap mBitmap)
	{
	    Mat mOriginalMat = convertToMat(mBitmap);
	    int mRatio = getRadio(mOriginalMat);
	    Size mSize = getImageFitSize(mOriginalMat, mRatio);

	    Mat resizedMat = resizeMat(mOriginalMat, mSize);
	    Mat colorMat = grayMat(resizedMat, mSize);
	    Mat blurMat = medianBlurMat(colorMat, mSize);
	    Mat thresholdMat = cannyEdgeMat(blurMat, mSize);

	    ArrayList<MatOfPoint> contourList = findContours(thresholdMat, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
	    double maxArea = 0.0;
	    int maxAreaIdx = -1;
	    Collections.sort(contourList, new Comparator<MatOfPoint>() {
	        @Override
	        public int compare(MatOfPoint lhs, MatOfPoint rhs)
	        {
	        return Double.valueOf(Imgproc.contourArea(rhs)).compareTo(Imgproc.contourArea(lhs));
	        }
	    });

	    ArrayList<MatOfPoint> contourListMax = new ArrayList<>();
	    for (int idx = 0; idx < contourList.size(); idx++)
	    {
	        MatOfPoint contour = contourList.get(idx);

	        MatOfPoint2f c2f = new MatOfPoint2f(contour.toArray());
	        MatOfPoint2f approx = new MatOfPoint2f();
	        double epsilon = Imgproc.arcLength(c2f, true);
	        Imgproc.approxPolyDP(c2f, approx, epsilon * 0.02, true);

	        Point[] points = approx.toArray();
	        MatOfPoint approxTemp = new MatOfPoint(approx.toArray());

	        if (points.length == 4 && Imgproc.isContourConvex(approxTemp) && maxArea < Imgproc.contourArea(approxTemp))
	        {
	            maxArea = Imgproc.contourArea(approxTemp);
	            maxAreaIdx = idx;
	            Point[] foundPoints = sortPoints(points);

	            contourListMax.add(approxTemp);

	            mPointFMap = new HashMap<>();
	            mPointFMap.put(0, new PointF((float) foundPoints[0].x + xGap, (float) foundPoints[0].y + yGap));
	            mPointFMap.put(1, new PointF((float) foundPoints[1].x + xGap, (float) foundPoints[1].y + yGap));
	            mPointFMap.put(2, new PointF((float) foundPoints[3].x + xGap, (float) foundPoints[3].y + yGap));
	            mPointFMap.put(3, new PointF((float) foundPoints[2].x + xGap, (float) foundPoints[2].y + yGap));
	            break;
	        }
	    }

	    Imgproc.drawContours(resizedMat, contourListMax, -1, new Scalar(255, 165, 0), 2);
	    showMatToImageView(resizedMat);
	}

	private Mat convertToMat(Bitmap bitmap)
	{
	    Mat mat = Imgcodecs.imread(mFilePath);// new Mat(bitmap.getWidth(), bitmap.getHeight(), CvType.CV_8UC1);
	    Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2RGB);
	    return mat;
	}

	private double getRadio(Mat mat)
	{
	    double ratio;
	    if (mat.size().width > mat.size().height)
	        ratio = mat.size().height / mMainLayout.getHeight();
	    else
	        ratio = mat.size().width / mMainLayout.getWidth();
	    return ratio;
	}

	private Size getImageFitSize(Mat mat, double ratio)
	{
	    int height = Double.valueOf(mat.size().height / ratio).intValue();
	    int width = Double.valueOf(mat.size().width / ratio).intValue();
	    return new Size(width, height);
	}

	private void showMatToImageView(Mat mat)
	{
	    final Bitmap bitmap = Bitmap.createBitmap(mat.width(), mat.height(), Bitmap.Config.ARGB_8888);
	    Utils.matToBitmap(mat, bitmap);
	    runOnUiThread(new Runnable()
	    {
	        @Override
	        public void run()
	        {
	            mSourceImageView.setImageBitmap(bitmap);
	            mProgressBar.setVisibility(View.GONE);
	        }
	    });
	}

	private Mat resizeMat(Mat mat, Size size)
	{
	    Mat resizedMat = new Mat(size, CvType.CV_8UC4);
	    Imgproc.resize(mat, resizedMat, size);
	    return resizedMat;
	}

	private Mat grayMat(Mat mat, Size size)
	{
	    Mat grayMat = new Mat(size, CvType.CV_8UC4);
	    Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_RGB2GRAY, 4);
	    return grayMat;
	}

	private Mat medianBlurMat(Mat mat, Size size)
	{
	    Mat blurMat = new Mat(size, CvType.CV_8UC4);
	    Imgproc.medianBlur(mat, blurMat, 3);
	    return blurMat;
	}

	private Mat cannyEdgeMat(Mat mat, Size size)
	{
	    if (thresholdVal <= 0)
	        thresholdVal = 200;
	    Mat cannyEdgeMat = new Mat(size, CvType.CV_8UC1);
	    Imgproc.Canny(mat, cannyEdgeMat, thresholdVal * 0.5, thresholdVal, 3, true);
	    return cannyEdgeMat;
	}

	private ArrayList<MatOfPoint> findContours(Mat mat, int retrievalMode, int approximationMode)
	{
	    ArrayList<MatOfPoint> contourList = new ArrayList<>();
	    Mat hierarchy = new Mat();
	    Imgproc.findContours(mat, contourList, hierarchy, retrievalMode, approximationMode);
	    hierarchy.release();

	    return contourList;
	}


}

/*
public Page recognizeTextBlocks(Path path) {
        log.info("TessBaseAPIGetComponentImages");
        File image = new File(path.toString());
        Leptonica leptInstance = Leptonica.INSTANCE;
        Pix pix = leptInstance.pixRead(image.getPath());
        Page blocks = new Page(pix.w,pix.h);        
        api.TessBaseAPIInit3(handle, datapath, language);
        api.TessBaseAPISetImage2(handle, pix);
        api.TessBaseAPISetPageSegMode(handle, TessPageSegMode.PSM_AUTO_OSD);
        PointerByReference pixa = null;
        PointerByReference blockids = null;
        Boxa boxes = api.TessBaseAPIGetComponentImages(handle, TessPageIteratorLevel.RIL_BLOCK, FALSE, pixa, blockids);
        int boxCount = leptInstance.boxaGetCount(boxes);
        for (int i = 0; i < boxCount; i++) {
            Box box = leptInstance.boxaGetBox(boxes, i, L_CLONE);
            if (box == null) {
                continue;
            }
            api.TessBaseAPISetRectangle(handle, box.x, box.y, box.w, box.h);
            Pointer utf8Text = api.TessBaseAPIGetUTF8Text(handle);
            String ocrResult = utf8Text.getString(0);
            Block block = null;
            if(ocrResult == null || (ocrResult.replace("\n", "").replace(" ","")).length() == 0){
                block = new ImageBlock(new Rectangle(box.x, box.y, box.w, box.h));
            }else{
                block = new TextBlock(new Rectangle(box.x, box.y, box.w, box.h), ocrResult); 
            }
            blocks.add(block);
            api.TessDeleteText(utf8Text);
            int conf = api.TessBaseAPIMeanTextConf(handle);
            log.debug(String.format("Box[%d]: x=%d, y=%d, w=%d, h=%d, confidence: %d, text: %s", i, box.x, box.y, box.w, box.h, conf, ocrResult));
        }

        //release Pix resource
        PointerByReference pRef = new PointerByReference();
        pRef.setValue(pix.getPointer());
        leptInstance.pixDestroy(pRef);

        return blocks;
    } * 
 */


*/
