package com.example.pLoc_samsung_opencv;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.annotation.NonNull;


import android.Manifest;
import android.content.ActivityNotFoundException;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.os.Bundle;
import android.util.Log;

import android.content.Intent;
import android.provider.MediaStore;
import android.view.SurfaceView;


import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.ORB;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
//import org.vlfeat.VLFeat;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;


import org.opencv.features2d.SIFT;

class Point{
    float []pointValue;
    static int vectorCount = 2;
    public Point(float []featureVector){
        int size = featureVector.length;
        if (size!=vectorCount){
            System.out.println("Vector count wrong!.");
            return;
        }
        pointValue = new float[size];
        for(int i=0;i<size;i++){
            pointValue[i] = featureVector[i];
        }
        return;
    }
    public void showPoint(){
        for(int i=0;i<vectorCount;i++){
            if(i!=0)System.out.print(",");
            System.out.print(pointValue[i]);
        }
        System.out.println("\n");
        return;
    }
    static float distance(Point a,Point b){
        float distance = 0;
        distance = (float) Math.hypot(a.pointValue[0] - b.pointValue[0], a.pointValue[1] - b.pointValue[1]);
        //for(int i=0;i<vectorCount;i++){
            //distance += (a.pointValue[i] - b.pointValue[i])*(a.pointValue[i] - b.pointValue[i]);
        //}
        //return Math.sqrt(distance);
        return distance;
    }
    static float pointSetDistance(Point []a, Point b){
        float distance = 0;
        for(int i =0;i<a.length;i++){
            distance += distance(a[i],b);
        }
        return distance;
    }
}



class FPSmanager{
    int setSize;
    int samplePointSize;
    Point[] pointList;
    float[] dists;
    ArrayList<Integer> samplePoint = new ArrayList<Integer>();
    ArrayList<Integer> leftPoint = new ArrayList<Integer>();

    public static int[] solve(ArrayList<Integer> nums)
    {
        int[] arr=new int[nums.size()];
        for(int i=0;i<nums.size();i++)
        {
            arr[i]=nums.get(i);
        }

        return arr;

    }

    public FPSmanager(int size,Point[] list,int sampleSize){
        setSize = size;
        samplePointSize = sampleSize;
        pointList = new Point[size];
        dists = new float[size];
        for(int i =0 ;i<size;i++){
            pointList[i] = list[i];
            dists[i] = Float.POSITIVE_INFINITY;
            leftPoint.add(i);

        }

    }

    public int[] FPSProcess(){
        Random r1 = new Random();
        int startIndex = r1.nextInt(setSize);
        samplePoint.add(startIndex);
        leftPoint.remove(startIndex);
        //System.out.println(startIndex);
        //for(int k = 0;k<leftPoint.size();k++){
            //System.out.print(leftPoint.get(k)+",");
        //}
        //System.out.println("");
        while(samplePoint.size() < samplePointSize){
            int curIndex = findFarhestPoint();
            samplePoint.add(new Integer(curIndex));
            leftPoint.remove(new Integer(curIndex));

            //System.out.println(curIndex);
            //for(int k = 0;k<leftPoint.size();k++){
                //System.out.print(leftPoint.get(k)+",");
            //}
            //System.out.println("");
        }

        return solve(samplePoint);

    }

    public int findFarhestPoint(){
        int index = -1;
        double maxlength = -1;
        for(int i=0;i<leftPoint.size();i++){
            float distance = 0;
            //System.out.println("haha"+leftPoint.get(i));
            int iIndex = leftPoint.get(i);
            //
            //for(int j =0;j<samplePoint.size();j++){
            distance=Point.distance(pointList[samplePoint.get(samplePoint.size()-1)],pointList[iIndex]);
            dists[iIndex] = Math.min(dists[iIndex], distance);
            //}
        }
        for(int i=0;i<leftPoint.size();i++) {
            int iIndex = leftPoint.get(i);
            float distance = dists[iIndex];
            if (distance > maxlength) {
                index = iIndex;
                maxlength = distance;
            }
        }
        return index;
    }


}


public class MainActivity extends AppCompatActivity {
    //private VLFeat vl;
    private Mat image;

    private int chunk_size = 600;

    private int all_feature_extraction_time = 0;
    private int all_mlp_time = 0;
    private int all_fps_time = 0;
    private int all_feature_selection = 0;
    private int image_num = 0;
    private int fps_image_num = 0;

    //private static final String TAG = "CameraActivity";
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 100;

    private String cameraId;
    private SurfaceView surfaceView;


    List<Long> time_record = new ArrayList<>();
    List<Long> mlp_record = new ArrayList<>();
    List<Long> fps_record = new ArrayList<>();
    List<Long> end2end_record = new ArrayList<>();



    private float[][] ret_kp;
    private float[][] ret_des;



    private Interpreter tflite;
    private static final String tf_TAG = "DigitClassifier";
    boolean load_result;

    //private SIFT sift = SIFT.create();
    private static final String TAG = "SIFT";

    //static {
        //System.loadLibrary("vlfeat");

    //}



    /**
     * Memory-map the model file in Assets.
     */
    private MappedByteBuffer loadModelFile(String model) throws IOException {
        AssetFileDescriptor fileDescriptor = getApplicationContext().getAssets().openFd(model + ".tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        //fileDescriptor.close();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // load infer model
    private void loadModel(String model) {
        try {

            Interpreter.Options options = new Interpreter.Options();
            CompatibilityList compatList = new CompatibilityList();
            //compatList.isDelegateSupportedOnThisDevice()
            GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
            GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
            NnApiDelegate nnApiDelegate = null;
            if(true){
                // if the device has a supported GPU, add the GPU delegate


                options.addDelegate(gpuDelegate);



                nnApiDelegate = new NnApiDelegate();
                options.addDelegate(nnApiDelegate);

            } else {
                // if the GPU is not supported, run on 4 threads
                options.setNumThreads(4);
            }
            load_result = true;
            tflite = new Interpreter(loadModelFile(model), options);
            tflite.resizeInput(0, new int[]{chunk_size, 128}, true);
            tflite.allocateTensors();

        } catch (IOException e) {
            Log.d(tf_TAG, model + " model load fail"+e);
            load_result = false;
            e.printStackTrace();
        }
    }

    private void init() {
        //测试一:获取asset下图片资源
        try {
            AssetManager assetManager = getAssets();
            InputStream is = assetManager.open("1.jpg");

            BitmapFactory.Options bmpFactoryOptions = new BitmapFactory.Options();
            bmpFactoryOptions.inPreferredConfig = Bitmap.Config.ARGB_8888;

            Bitmap bitmap = BitmapFactory.decodeStream(is, null, bmpFactoryOptions);
            image = new Mat();
            Utils.bitmapToMat(bitmap, image);
            Log.d(TAG, "loadedImage: " + "chans: " + image.channels()
                    + ", (" + image.width() + ", " + image.height() + ")");


            //Bitmap bitmap = BitmapFactory.decodeStream(is);

            if (bitmap != null) {
                System.out.println("测试一:width=" + bitmap.getWidth() + " ,height=" + bitmap.getHeight());
            } else {
                System.out.println("bitmap == null");
            }
        } catch (Exception e) {
            System.out.println("异常信息:" + e.toString());
        }

        System.out.println("======================");
    }
    public static <T extends Number> int[] asArray(final T... a) {
        int[] b = new int[a.length];
        for (int i = 0; i < b.length; i++) {
            b[i] = a[i].intValue();
        }
        return b;
    }
    public static int[] argsort(final float[] a, final boolean ascending) {
        Integer[] indexes = new Integer[a.length];
        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = i;
        }
        Arrays.sort(indexes, new Comparator<Integer>() {
            @Override
            public int compare(final Integer i1, final Integer i2) {
                return (ascending ? 1 : -1) * Float.compare(a[i1], a[i2]);
            }
        });
        return asArray(indexes);
    }


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i("OpenCV", "OpenCV loaded successfully");
                    image=new Mat();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };


    private void dispatchTakePictureIntent() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 2);
        } else {
            startCameraIntent();
        }
    }

//    @Override
//    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
//        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
//        if (requestCode == 2) {
//            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
//                startCameraIntent();
//            } else {
//                // Permission was not granted, handle appropriately
//                Log.d(TAG, "not granted");
//            }
//        }
//    }

    private void startCameraIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        try {
            //startActivityForResult(takePictureIntent, 1);
            startActivity(takePictureIntent);
        } catch (ActivityNotFoundException e) {
            // Display error state to the user
            Log.d(TAG, "Capture Failed");
        }
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 1 && resultCode == RESULT_OK) {
            Bundle extras = data.getExtras();
            Bitmap ori_bitmap = (Bitmap) extras.get("data");
            // Use the bitmap as you need

            int original_w = ori_bitmap.getWidth();
            int original_h = ori_bitmap.getHeight();

            float scale = (float) (1024.0 / Math.max(original_w, original_h));
            //float scale = 1.0F;
            int newWidth = (int) (scale * original_w);
            int newHeight = (int) (scale * original_h);

            Bitmap bitmap = Bitmap.createScaledBitmap(ori_bitmap, newWidth, newHeight, true);
            image = new Mat();
            Utils.bitmapToMat(bitmap, image);
            Log.d(TAG, "Camera Image: " + "chans: " + image.channels()
                    + ", (" + image.width() + ", " + image.height() + ")");
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            openCamera();
        } else {
            Log.e(TAG, "Camera permission was not granted.");
        }
    }

    private void openCamera() {
        CameraManager manager = (CameraManager) getSystemService(CAMERA_SERVICE);
        try {
            cameraId = manager.getCameraIdList()[0];
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
                return;
            }
            manager.openCamera(cameraId, new CameraDevice.StateCallback() {
                @Override
                public void onOpened(@NonNull CameraDevice camera) {
                    // Camera is opened, you can start preview here
                    Log.d(TAG, "Camera opened");
                }

                @Override
                public void onDisconnected(@NonNull CameraDevice camera) {
                    // Camera got disconnected
                    Log.d(TAG, "Camera disconnected");
                }

                @Override
                public void onError(@NonNull CameraDevice camera, int error) {
                    // Error occurred while opening the camera
                    Log.e(TAG, "Error opening camera: " + error);
                }
            }, null);
        } catch (CameraAccessException e) {
            Log.e(TAG, "Failed to open camera", e);
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);



        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_10, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!" + this.getClass().getName());
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

        AssetManager assetManager = getAssets();
        //vl = new VLFeat();
        loadModel("converted_model_opencv_4096_2048_50_float");
        //loadModel("converted_model");
        SIFT sift = SIFT.create(0,1,0.03);

        /*
        ORB orb = ORB.create();

        BFMatcher matcher = BFMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING, false);

        try {
            InputStream ref = assetManager.open("temp_images/ref.jpg");
            BitmapFactory.Options bmpFactoryOptions_ref = new BitmapFactory.Options();
            bmpFactoryOptions_ref.inPreferredConfig = Bitmap.Config.ARGB_8888;

            Bitmap ref_bitmap = BitmapFactory.decodeStream(ref, null, bmpFactoryOptions_ref);
            int original_refw = ref_bitmap.getWidth();
            int original_refh = ref_bitmap.getHeight();

            float scale = (float) (1280.0 / Math.max(original_refw, original_refh));
            //float scale = 1.0F;
            //int newWidth = (int) (scale * original_w);
            //int newHeight = (int) (scale * original_h);
            int newWidth_ref = 768;
            int newHeight_ref = 1024;
            Bitmap bitmap_ref = Bitmap.createScaledBitmap(ref_bitmap, newWidth_ref, newHeight_ref, true);
            Mat image_ref = new Mat();
            Utils.bitmapToMat(bitmap_ref, image_ref);
            Log.d(TAG, "loadedImage: " + "chans: " + image_ref.channels()
                    + ", (" + image_ref.width() + ", " + image_ref.height() + ")");


            Mat image_ref_gray = new Mat(image_ref.size(), image_ref.type());
            Imgproc.cvtColor(image_ref, image_ref_gray, Imgproc.COLOR_RGBA2GRAY);
//Opencv sift extraction

            long startTime_orb_ref = System.currentTimeMillis();
            MatOfKeyPoint keypoints_orb_ref = new MatOfKeyPoint();
            Mat descriptors_orb_ref = new Mat();

            orb.detectAndCompute(image_ref_gray, new Mat(), keypoints_orb_ref, descriptors_orb_ref);
            //vl.opencv_extractsift(image_gray.getNativeObjAddr());
            long estimatedTime_orb_ref = System.currentTimeMillis() - startTime_orb_ref;

            Log.d(TAG, "time of extracted features orv: " + estimatedTime_orb_ref);
            Log.d(TAG, "number of extracted features orb: " + descriptors_orb_ref.size());



            String[] getImages = assetManager.list("temp_images");
            for(String imgName : getImages){
                long end2end_startTime = System.currentTimeMillis();
                image_num += 1;
                Log.d(TAG,"IMAGE NAME----->ref.jpg");

                InputStream is = assetManager.open("temp_images/ref.jpg");

                BitmapFactory.Options bmpFactoryOptions = new BitmapFactory.Options();
                bmpFactoryOptions.inPreferredConfig = Bitmap.Config.ARGB_8888;

                Bitmap ori_bitmap = BitmapFactory.decodeStream(is, null, bmpFactoryOptions);
                long startTime_orb = System.currentTimeMillis();
                //int original_w = ori_bitmap.getWidth();
                //int original_h = ori_bitmap.getHeight();

                //float scale = (float) (1280.0 / Math.max(original_w, original_h));
                //float scale = 1.0F;
                //int newWidth = (int) (scale * original_w);
                //int newHeight = (int) (scale * original_h);
                int newWidth = 2160;
                int newHeight = 3840;
                Bitmap bitmap = Bitmap.createScaledBitmap(ori_bitmap, newWidth, newHeight, true);
                image = new Mat();
                Utils.bitmapToMat(bitmap, image);
                //Log.d(TAG, "loadedImage: " + "chans: " + image.channels()
                        //+ ", (" + image.width() + ", " + image.height() + ")");


                Mat image_gray = new Mat(image.size(), image.type());
                Imgproc.cvtColor(image, image_gray, Imgproc.COLOR_RGBA2GRAY);

                image = null;



                //Opencv sift extraction

                MatOfKeyPoint keypoints_orb = new MatOfKeyPoint();
                Mat descriptors_orb = new Mat();

                orb.detectAndCompute(image_gray, new Mat(), keypoints_orb, descriptors_orb);
                //vl.opencv_extractsift(image_gray.getNativeObjAddr())


                List<MatOfDMatch> matches = new ArrayList<MatOfDMatch>();
                matcher.knnMatch(descriptors_orb, descriptors_orb_ref, matches, 2);

                long estimatedTime_orb = System.currentTimeMillis() - startTime_orb;

                Log.d(TAG, "time of extracted features orv: " + estimatedTime_orb);
                //Log.d(TAG, "number of extracted features orb: " + descriptors_orb.size());





                TimeUnit.SECONDS.sleep(1);
//------------------------------------------------------------------------------------------------------------------


            }
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }*/


        //dispatchTakePictureIntent();

        //Nan 1010
        surfaceView = new SurfaceView(this);
        setContentView(surfaceView);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
        } else {
            openCamera();
        }


//
//        try {
//            String[] getImages = assetManager.list("Myimages");
//            while (true){
//                for(String imgName : getImages){
//
//                    long end2end_startTime = System.currentTimeMillis();
//                    image_num += 1;
//                    Log.d(TAG,"IMAGE NAME----->" + imgName);
//
//                    InputStream is = assetManager.open("Myimages/"+imgName);
//
//                    BitmapFactory.Options bmpFactoryOptions = new BitmapFactory.Options();
//                    bmpFactoryOptions.inPreferredConfig = Bitmap.Config.ARGB_8888;
//
//                    Bitmap ori_bitmap = BitmapFactory.decodeStream(is, null, bmpFactoryOptions);
//
//                    int original_w = ori_bitmap.getWidth();
//                    int original_h = ori_bitmap.getHeight();
//
//                    float scale = (float) (1024.0 / Math.max(original_w, original_h));
//                    //float scale = 1.0F;
//                    int newWidth = (int) (scale * original_w);
//                    int newHeight = (int) (scale * original_h);
//
//                    Bitmap bitmap = Bitmap.createScaledBitmap(ori_bitmap, newWidth, newHeight, true);
//                    image = new Mat();
//                    Utils.bitmapToMat(bitmap, image);
//                    Log.d(TAG, "loadedImage: " + "chans: " + image.channels()
//                            + ", (" + image.width() + ", " + image.height() + ")");
//
//
//                    Mat image_gray = new Mat(image.size(), image.type());
//                    Imgproc.cvtColor(image, image_gray, Imgproc.COLOR_RGBA2GRAY);
//
//                    image = null;
//
//
//                /*
//                //Opencv sift extraction
//                long startTime_orb = System.currentTimeMillis();
//                MatOfKeyPoint keypoints_orb = new MatOfKeyPoint();
//                Mat descriptors_orb = new Mat();
//
//                sift.detectAndCompute(image_gray, new Mat(), keypoints_orb, descriptors_orb);
//                //vl.opencv_extractsift(image_gray.getNativeObjAddr());
//                long estimatedTime_orb = System.currentTimeMillis() - startTime_orb;
//
//                Log.d(TAG, "time of extracted features orv: " + estimatedTime_orb);
//                Log.d(TAG, "number of extracted features orb: " + descriptors_orb.size());*/
//
//                    long startTime = System.currentTimeMillis();
//                    MatOfKeyPoint localKeypoint = new MatOfKeyPoint();
//                    Mat localDescriptor = new Mat();
//
//                    sift.detectAndCompute(image_gray, new Mat(), localKeypoint, localDescriptor);
//                    //vl.opencv_extractsift(image_gray.getNativeObjAddr());
//                    long estimatedTime = System.currentTimeMillis() - startTime;
//
//                    Log.d(TAG, "time of extracted features opencv: " + estimatedTime);
//                    Log.d(TAG, "number of extracted features opencv: " + localDescriptor.size());
//
//                    all_feature_extraction_time += estimatedTime;
//                    time_record.add(estimatedTime);
//
//                    long startTime_predict_slice = System.currentTimeMillis();
//                    float[][] inFloat = new float[localDescriptor.rows()][localDescriptor.cols()];
//                /*
//                for (int m = 0; m < localDescriptor.rows(); m++){
//                    for (int n = 0; n < localDescriptor.cols(); n++) {
//                        inFloat[m][n] = (float) ((float) localDescriptor.get(m, n)[0] / 255.0);
//                        //inFloat[m][n] = (float) ((float) localDescriptor.get(m, n)[0]);
//                    }
//                }*/
//                    int rows = localDescriptor.rows();
//                    int cols = localDescriptor.cols();
//                    float[] buffer = new float[cols];
//
//                    for (int i = 0; i < rows; i++) {
//                        localDescriptor.get(i, 0, buffer);
//                        for (int j = 0; j < cols; j++) {
//                            inFloat[i][j] = (float) (buffer[j] / 255.0);
//                        }
//                    }
//
//                    long estimatedTime_tt2 = System.currentTimeMillis() - startTime_predict_slice;
//                    Log.d(TAG, "time for preprocess descriptors: " + estimatedTime_tt2);
//
//                    long startTime_random = System.currentTimeMillis();
//
//
//                    // Calculate number of rows to select
//                    int numSelectRows = (int) Math.ceil(rows * 0.10);  // use Math.ceil to always round up
//
//                    // Generate a list of indices
//                    List<Integer> indices = new ArrayList<>();
//                    for (int i = 0; i < rows; i++) {
//                        indices.add(i);
//                    }
//
//                    // Shuffle the list and select the first numSelectRows indices
//                    Collections.shuffle(indices);
//                    List<Integer> selectedIndices = indices.subList(0, numSelectRows);
//
//                    // Create the outFloat array
//                    float[][] outFloat = new float[numSelectRows][cols];
//                    for (int i = 0; i < numSelectRows; i++) {
//                        outFloat[i] = inFloat[selectedIndices.get(i)];
//                    }
//
//                    long estimatedTime_random = System.currentTimeMillis() - startTime_random;
//                    Log.d(TAG, "time for random selection: " + estimatedTime_random);
//
//
////
////                    float[][] yy = new float[inFloat.length][1];
////                    for (int ii = 0; ii < (inFloat.length-1) / chunk_size + 1; ii++){
////                        float[][] temp_in = new float[chunk_size][128];
////                        for (int j = ii * chunk_size; j < Math.min((ii + 1)* chunk_size, inFloat.length); j++){
////                            temp_in[j - ii * chunk_size] = inFloat[j];
////                        }
////                        float[][] temp_y = new float[chunk_size][1];
////
////                        long startTime_tt = System.currentTimeMillis();
////                        tflite.run(temp_in, temp_y);
////                        long estimatedTime_tt = System.currentTimeMillis() - startTime_tt;
////                        Log.d(TAG, "time for new MLP prediction: " + estimatedTime_tt);
////                        for (int j = 0; j < chunk_size; j++){
////                            if (ii*chunk_size + j < inFloat.length) {
////                                yy[ii*chunk_size + j] = temp_y[j];
////                            }
////
////                        }
////
////                    }
////                    long estimatedTime_predict_slice = System.currentTimeMillis() - startTime_predict_slice;
////                    Log.d(TAG, "time for new MLP prediction: " + estimatedTime_predict_slice);
////                    all_mlp_time += estimatedTime_predict_slice;
////                    mlp_record.add(estimatedTime_predict_slice);
////
////
////
////                    long startTime_feature_selection = System.currentTimeMillis();
////                    float[] prediction_confidence = new float[yy.length];
////                    for (int ii = 0; ii < yy.length; ii++){
////                        prediction_confidence[ii] = yy[ii][0];
////                    }
////
////
////                    int target_num = prediction_confidence.length / 10;
////                    int[] sort_index = argsort(prediction_confidence, false);
////                    float flag_num = prediction_confidence[sort_index[Math.min(prediction_confidence.length, target_num)-1]];
////                    Log.d(TAG, "flag_num is: " + flag_num);
////
////                    //int target_num = sort_index.length/10;
////
////
////                    if (flag_num > 0.5) {
////                        fps_image_num += 1;
////
////
////                        float[][] thresh_select_kp;
////                        float[][] thresh_select_des;
////
////                        int thresh_select_num = 0;
////                        for (int feature_idx = 0; feature_idx < prediction_confidence.length; feature_idx++) {
////                            if (prediction_confidence[feature_idx] > 0.5) {
////                                thresh_select_num += 1;
////                            }
////                        }
////                        Log.d(TAG, "thresh_select_num: " + thresh_select_num);
////
////                        thresh_select_kp = new float[thresh_select_num][2];
////                        thresh_select_des = new float[thresh_select_num][128];
////
////                        KeyPoint[] keyPointsArr = localKeypoint.toArray();
////                        int thresh_select_idx = 0;
////                        for (int feature_idx = 0; feature_idx < prediction_confidence.length; feature_idx++) {
////                            if (prediction_confidence[feature_idx] > 0.5) {
////                                thresh_select_kp[thresh_select_idx][0] = (float) keyPointsArr[feature_idx].pt.x;
////                                thresh_select_kp[thresh_select_idx][1] = (float) keyPointsArr[feature_idx].pt.y;
////                                float[] new_buffer = new float[128];
////                                localDescriptor.get(feature_idx, 0, buffer);
////                                for (int n = 0; n < localDescriptor.cols(); n++) {
////                                    thresh_select_des[thresh_select_idx][n] = new_buffer[n];
////                                    //thresh_select_des[thresh_select_idx][n] = (float) localDescriptor.get(feature_idx, n)[0];
////                                }
////                                thresh_select_idx += 1;
////                            }
////
////                        }
////
////
////
////
////
////
////                        long startTime_fps = System.currentTimeMillis();
////                        //test FPS on all points
////                        Point[] test = new Point[thresh_select_kp.length];
////                        for (int point_id = 0; point_id < thresh_select_kp.length; point_id++) {
////                            float[] tmpVector = new float[2];
////                            tmpVector[0] = thresh_select_kp[point_id][0];
////                            tmpVector[1] = thresh_select_kp[point_id][1];
////                            test[point_id] = new Point(tmpVector);
////                        }
////
////                        target_num = Math.min(thresh_select_kp.length, target_num);
////
////                        FPSmanager fpsManager = new FPSmanager(thresh_select_kp.length, test,  target_num);
////                        int[] p = fpsManager.FPSProcess();
////                        long estimatedTime_fps = System.currentTimeMillis() - startTime_fps;
////                        Log.d(TAG, "time for FPS prediction: " + estimatedTime_fps);
////                        all_fps_time += estimatedTime_fps;
////
////                        ret_kp = new float[target_num][2];
////                        ret_des = new float[target_num][128];
////
////                        for (int i = 0; i < target_num; i++){
////                            ret_kp[i] = thresh_select_kp[p[i]];
////                            ret_des[i] = thresh_select_des[p[i]];
////
////                            float[] new_buffer = new float[128];
////                            localDescriptor.get(p[i], 0, buffer);
////                            for (int n = 0; n < localDescriptor.cols(); n++) {
////                                ret_des[i][n] = new_buffer[n];
////                                //ret_des[i][n] = (float) localDescriptor.get(p[i], n)[0];
////                            }
////                        }
////
////                    /*for (int i = 0; i < target_num; i++){
////                        System.out.print(ret_kp[i][0] + " ");
////                        System.out.print(ret_kp[i][1]);
////                        System.out.println();
////                    }*/
////
////
////
////                    }else{
////                        target_num = Math.min(target_num, localKeypoint.rows());
////
////                        int[] p = Arrays.copyOfRange(sort_index, 0, target_num);
////
////                        ret_kp = new float[target_num][2];
////                        ret_des = new float[target_num][128];
////
////                        KeyPoint[] keyPointsArr = localKeypoint.toArray();
////                        for (int i = 0; i < target_num; i++){
////                            ret_kp[i][0] = (float) keyPointsArr[p[i]].pt.x;
////                            ret_kp[i][1] = (float) keyPointsArr[p[i]].pt.y;
////
////                            float[] new_buffer = new float[128];
////                            localDescriptor.get(p[i], 0, buffer);
////                            for (int n = 0; n < localDescriptor.cols(); n++) {
////                                ret_des[i][n] = new_buffer[n];
////                                //ret_des[i][n] = (float) localDescriptor.get(p[i], n)[0];
////                            }
////
////                            //for (int n = 0; n < localDescriptor.cols(); n++) {
////                            //    ret_des[i][n] = (float) localDescriptor.get(p[i], n)[0];
////                            //}
////                        }
////                    /*for (int i = 0; i < target_num; i++){
////                        System.out.print(ret_kp[i][0] + " ");
////                        System.out.print(ret_kp[i][1]);
////                        System.out.println();
////                    }*/
////                    }
////
////                    Log.d(TAG, "Num of selected features: " + ret_kp.length);
////
////                    long estimatedTime_feature_selection= System.currentTimeMillis() - startTime_feature_selection;
////                    Log.d(TAG, "time for Feature Selection: " + estimatedTime_feature_selection);
////                    all_feature_selection += estimatedTime_feature_selection;
////                    fps_record.add(estimatedTime_feature_selection);
//
//                    long end2end_estimatedTime = System.currentTimeMillis() - end2end_startTime;
//
//                    end2end_record.add(end2end_estimatedTime);
//                    Log.d(TAG, "time for end to end: " + end2end_estimatedTime);
//
//
//
//                    TimeUnit.SECONDS.sleep(1);
//                    //Thread.sleep(15000);
//
//
//                }
//            }
//
//
//        } catch (IOException | InterruptedException e) {
//            e.printStackTrace();
//        }
////        } catch (IOException e) {
////            e.printStackTrace();
////        }
//
//        Log.d(TAG, "time: "+time_record.subList(0, time_record.size()/2));
//        Log.d(TAG, "time: "+time_record.subList(time_record.size()/2, time_record.size()));
//        Log.d(TAG, "mlp: "+mlp_record.subList(0, mlp_record.size()/2));
//        Log.d(TAG, "mlp: "+mlp_record.subList(mlp_record.size()/2, mlp_record.size()));
//        Log.d(TAG, "hybrid: "+fps_record.subList(0, fps_record.size()/2));
//        Log.d(TAG, "hybrid: "+fps_record.subList(fps_record.size()/2, fps_record.size()));
//        Log.d(TAG, "end to end: "+end2end_record.subList(0, end2end_record.size()/2));
//        Log.d(TAG, "end to end: "+end2end_record.subList(end2end_record.size()/2, end2end_record.size()));
//
//
//
//        Log.d(TAG, "All time for extracted features opencv: " + all_feature_extraction_time);
//        Log.d(TAG, "All time for MLP: " + all_mlp_time);
//        Log.d(TAG, "All time for FPS: " + all_fps_time);
//        Log.d(TAG, "All time for Feature Selection: " + all_feature_selection);
//
//        Log.d(TAG, "Average time for extracted features opencv: " + all_feature_extraction_time / image_num);
//        Log.d(TAG, "Average time for MLP: " + all_mlp_time / image_num);
//        Log.d(TAG, "Average time for FPS: " + all_fps_time / fps_image_num);
//        Log.d(TAG, "Average time for Feature Selection: " + all_feature_selection / image_num);

    }

    @Override
    protected void onResume() {
        super.onResume();


        AssetManager assetManager = getAssets();
        SIFT sift = SIFT.create(0,1,0.03);
        ORB orb = ORB.create(5000);

        try {
            String[] getImages = assetManager.list("temp_images");
            while (true){
                for(String imgName : getImages){

                    long end2end_startTime = System.currentTimeMillis();
                    image_num += 1;
                    Log.d(TAG,"IMAGE NAME----->" + imgName);

                    InputStream is = assetManager.open("temp_images/"+imgName);

                    BitmapFactory.Options bmpFactoryOptions = new BitmapFactory.Options();
                    bmpFactoryOptions.inPreferredConfig = Bitmap.Config.ARGB_8888;

                    Bitmap ori_bitmap = BitmapFactory.decodeStream(is, null, bmpFactoryOptions);

                    int original_w = ori_bitmap.getWidth();
                    int original_h = ori_bitmap.getHeight();

                    float scale = (float) (1024.0 / Math.max(original_w, original_h));
                    //float scale = 1.0F;
                    int newWidth = (int) (scale * original_w);
                    int newHeight = (int) (scale * original_h);

                    Bitmap bitmap = Bitmap.createScaledBitmap(ori_bitmap, newWidth, newHeight, true);
                    image = new Mat();
                    Utils.bitmapToMat(bitmap, image);
                    Log.d(TAG, "loadedImage: " + "chans: " + image.channels()
                            + ", (" + image.width() + ", " + image.height() + ")");


                    Mat image_gray = new Mat(image.size(), image.type());
                    Imgproc.cvtColor(image, image_gray, Imgproc.COLOR_RGBA2GRAY);

                    image = null;


                /*
                //Opencv sift extraction
                long startTime_orb = System.currentTimeMillis();
                MatOfKeyPoint keypoints_orb = new MatOfKeyPoint();
                Mat descriptors_orb = new Mat();

                sift.detectAndCompute(image_gray, new Mat(), keypoints_orb, descriptors_orb);
                //vl.opencv_extractsift(image_gray.getNativeObjAddr());
                long estimatedTime_orb = System.currentTimeMillis() - startTime_orb;

                Log.d(TAG, "time of extracted features orv: " + estimatedTime_orb);
                Log.d(TAG, "number of extracted features orb: " + descriptors_orb.size());*/

                    long startTime_orb_ref = System.currentTimeMillis();
                    MatOfKeyPoint keypoints_orb_ref = new MatOfKeyPoint();
                    Mat descriptors_orb_ref = new Mat();

                    orb.detectAndCompute(image_gray, new Mat(), keypoints_orb_ref, descriptors_orb_ref);
                    //vl.opencv_extractsift(image_gray.getNativeObjAddr());
                    long estimatedTime_orb_ref = System.currentTimeMillis() - startTime_orb_ref;

                    Log.d(TAG, "time of extracted features orv: " + estimatedTime_orb_ref);
                    Log.d(TAG, "number of extracted features orb: " + descriptors_orb_ref.size());

                    long startTime = System.currentTimeMillis();
                    MatOfKeyPoint localKeypoint = new MatOfKeyPoint();
                    Mat localDescriptor = new Mat();

                    sift.detectAndCompute(image_gray, new Mat(), localKeypoint, localDescriptor);
                    //vl.opencv_extractsift(image_gray.getNativeObjAddr());
                    long estimatedTime = System.currentTimeMillis() - startTime;

                    Log.d(TAG, "time of extracted features opencv: " + estimatedTime);
                    Log.d(TAG, "number of extracted features opencv: " + localDescriptor.size());

                    all_feature_extraction_time += estimatedTime;
                    time_record.add(estimatedTime);

                    long startTime_predict_slice = System.currentTimeMillis();
                    float[][] inFloat = new float[localDescriptor.rows()][localDescriptor.cols()];
                /*
                for (int m = 0; m < localDescriptor.rows(); m++){
                    for (int n = 0; n < localDescriptor.cols(); n++) {
                        inFloat[m][n] = (float) ((float) localDescriptor.get(m, n)[0] / 255.0);
                        //inFloat[m][n] = (float) ((float) localDescriptor.get(m, n)[0]);
                    }
                }*/
                    int rows = localDescriptor.rows();
                    int cols = localDescriptor.cols();
                    float[] buffer = new float[cols];

                    for (int i = 0; i < rows; i++) {
                        localDescriptor.get(i, 0, buffer);
                        for (int j = 0; j < cols; j++) {
                            inFloat[i][j] = (float) (buffer[j] / 255.0);
                        }
                    }

                    long estimatedTime_tt2 = System.currentTimeMillis() - startTime_predict_slice;
                    Log.d(TAG, "time for preprocess descriptors: " + estimatedTime_tt2);
//
//                    long startTime_random = System.currentTimeMillis();
//
//
//                    // Calculate number of rows to select
//                    int numSelectRows = (int) Math.ceil(rows * 0.10);  // use Math.ceil to always round up
//
//                    // Generate a list of indices
//                    List<Integer> indices = new ArrayList<>();
//                    for (int i = 0; i < rows; i++) {
//                        indices.add(i);
//                    }
//
//                    // Shuffle the list and select the first numSelectRows indices
//                    Collections.shuffle(indices);
//                    List<Integer> selectedIndices = indices.subList(0, numSelectRows);
//
//                    // Create the outFloat array
//                    float[][] outFloat = new float[numSelectRows][cols];
//                    for (int i = 0; i < numSelectRows; i++) {
//                        outFloat[i] = inFloat[selectedIndices.get(i)];
//                    }
//
//                    long estimatedTime_random = System.currentTimeMillis() - startTime_random;
//                    Log.d(TAG, "time for random selection: " + estimatedTime_random);



                    float[][] yy = new float[inFloat.length][1];
                    for (int ii = 0; ii < (inFloat.length-1) / chunk_size + 1; ii++){
                        float[][] temp_in = new float[chunk_size][128];
                        for (int j = ii * chunk_size; j < Math.min((ii + 1)* chunk_size, inFloat.length); j++){
                            temp_in[j - ii * chunk_size] = inFloat[j];
                        }
                        float[][] temp_y = new float[chunk_size][1];

                        long startTime_tt = System.currentTimeMillis();
                        tflite.run(temp_in, temp_y);
                        long estimatedTime_tt = System.currentTimeMillis() - startTime_tt;
                        Log.d(TAG, "time for new MLP prediction: " + estimatedTime_tt);
                        for (int j = 0; j < chunk_size; j++){
                            if (ii*chunk_size + j < inFloat.length) {
                                yy[ii*chunk_size + j] = temp_y[j];
                            }

                        }

                    }
                    long estimatedTime_predict_slice = System.currentTimeMillis() - startTime_predict_slice;
                    Log.d(TAG, "time for new MLP prediction: " + estimatedTime_predict_slice);
                    all_mlp_time += estimatedTime_predict_slice;
                    mlp_record.add(estimatedTime_predict_slice);



                    long startTime_feature_selection = System.currentTimeMillis();
                    float[] prediction_confidence = new float[yy.length];
                    for (int ii = 0; ii < yy.length; ii++){
                        prediction_confidence[ii] = yy[ii][0];
                    }


                    int target_num = prediction_confidence.length / 10;
                    int[] sort_index = argsort(prediction_confidence, false);
                    float flag_num = prediction_confidence[sort_index[Math.min(prediction_confidence.length, target_num)-1]];
                    Log.d(TAG, "flag_num is: " + flag_num);

                    //int target_num = sort_index.length/10;


                    if (flag_num > 0.5) {
                        fps_image_num += 1;


                        float[][] thresh_select_kp;
                        float[][] thresh_select_des;

                        int thresh_select_num = 0;
                        for (int feature_idx = 0; feature_idx < prediction_confidence.length; feature_idx++) {
                            if (prediction_confidence[feature_idx] > 0.5) {
                                thresh_select_num += 1;
                            }
                        }
                        Log.d(TAG, "thresh_select_num: " + thresh_select_num);

                        thresh_select_kp = new float[thresh_select_num][2];
                        thresh_select_des = new float[thresh_select_num][128];

                        KeyPoint[] keyPointsArr = localKeypoint.toArray();
                        int thresh_select_idx = 0;
                        for (int feature_idx = 0; feature_idx < prediction_confidence.length; feature_idx++) {
                            if (prediction_confidence[feature_idx] > 0.5) {
                                thresh_select_kp[thresh_select_idx][0] = (float) keyPointsArr[feature_idx].pt.x;
                                thresh_select_kp[thresh_select_idx][1] = (float) keyPointsArr[feature_idx].pt.y;
                                float[] new_buffer = new float[128];
                                localDescriptor.get(feature_idx, 0, buffer);
                                for (int n = 0; n < localDescriptor.cols(); n++) {
                                    thresh_select_des[thresh_select_idx][n] = new_buffer[n];
                                    //thresh_select_des[thresh_select_idx][n] = (float) localDescriptor.get(feature_idx, n)[0];
                                }
                                thresh_select_idx += 1;
                            }

                        }






                        long startTime_fps = System.currentTimeMillis();
                        //test FPS on all points
                        Point[] test = new Point[thresh_select_kp.length];
                        for (int point_id = 0; point_id < thresh_select_kp.length; point_id++) {
                            float[] tmpVector = new float[2];
                            tmpVector[0] = thresh_select_kp[point_id][0];
                            tmpVector[1] = thresh_select_kp[point_id][1];
                            test[point_id] = new Point(tmpVector);
                        }

                        target_num = Math.min(thresh_select_kp.length, target_num);

                        FPSmanager fpsManager = new FPSmanager(thresh_select_kp.length, test,  target_num);
                        int[] p = fpsManager.FPSProcess();
                        long estimatedTime_fps = System.currentTimeMillis() - startTime_fps;
                        Log.d(TAG, "time for FPS prediction: " + estimatedTime_fps);
                        all_fps_time += estimatedTime_fps;

                        ret_kp = new float[target_num][2];
                        ret_des = new float[target_num][128];

                        for (int i = 0; i < target_num; i++){
                            ret_kp[i] = thresh_select_kp[p[i]];
                            ret_des[i] = thresh_select_des[p[i]];

                            float[] new_buffer = new float[128];
                            localDescriptor.get(p[i], 0, buffer);
                            for (int n = 0; n < localDescriptor.cols(); n++) {
                                ret_des[i][n] = new_buffer[n];
                                //ret_des[i][n] = (float) localDescriptor.get(p[i], n)[0];
                            }
                        }

                    /*for (int i = 0; i < target_num; i++){
                        System.out.print(ret_kp[i][0] + " ");
                        System.out.print(ret_kp[i][1]);
                        System.out.println();
                    }*/



                    }else{
                        target_num = Math.min(target_num, localKeypoint.rows());

                        int[] p = Arrays.copyOfRange(sort_index, 0, target_num);

                        ret_kp = new float[target_num][2];
                        ret_des = new float[target_num][128];

                        KeyPoint[] keyPointsArr = localKeypoint.toArray();
                        for (int i = 0; i < target_num; i++){
                            ret_kp[i][0] = (float) keyPointsArr[p[i]].pt.x;
                            ret_kp[i][1] = (float) keyPointsArr[p[i]].pt.y;

                            float[] new_buffer = new float[128];
                            localDescriptor.get(p[i], 0, buffer);
                            for (int n = 0; n < localDescriptor.cols(); n++) {
                                ret_des[i][n] = new_buffer[n];
                                //ret_des[i][n] = (float) localDescriptor.get(p[i], n)[0];
                            }

                            //for (int n = 0; n < localDescriptor.cols(); n++) {
                            //    ret_des[i][n] = (float) localDescriptor.get(p[i], n)[0];
                            //}
                        }
                    /*for (int i = 0; i < target_num; i++){
                        System.out.print(ret_kp[i][0] + " ");
                        System.out.print(ret_kp[i][1]);
                        System.out.println();
                    }*/
                    }

                    Log.d(TAG, "Num of selected features: " + ret_kp.length);

                    long estimatedTime_feature_selection= System.currentTimeMillis() - startTime_feature_selection;
                    Log.d(TAG, "time for Feature Selection: " + estimatedTime_feature_selection);
                    all_feature_selection += estimatedTime_feature_selection;
                    fps_record.add(estimatedTime_feature_selection);

                    long end2end_estimatedTime = System.currentTimeMillis() - end2end_startTime;

                    end2end_record.add(end2end_estimatedTime);
                    Log.d(TAG, "time for end to end: " + end2end_estimatedTime);



                    //TimeUnit.SECONDS.sleep(1);
                    //Thread.sleep(15000);
                    TimeUnit.MILLISECONDS.sleep(1400);


                }
            }


        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

        Log.d(TAG, "time: "+time_record.subList(0, time_record.size()/2));
        Log.d(TAG, "time: "+time_record.subList(time_record.size()/2, time_record.size()));
        Log.d(TAG, "mlp: "+mlp_record.subList(0, mlp_record.size()/2));
        Log.d(TAG, "mlp: "+mlp_record.subList(mlp_record.size()/2, mlp_record.size()));
        Log.d(TAG, "hybrid: "+fps_record.subList(0, fps_record.size()/2));
        Log.d(TAG, "hybrid: "+fps_record.subList(fps_record.size()/2, fps_record.size()));
        Log.d(TAG, "end to end: "+end2end_record.subList(0, end2end_record.size()/2));
        Log.d(TAG, "end to end: "+end2end_record.subList(end2end_record.size()/2, end2end_record.size()));



        Log.d(TAG, "All time for extracted features opencv: " + all_feature_extraction_time);
        Log.d(TAG, "All time for MLP: " + all_mlp_time);
        Log.d(TAG, "All time for FPS: " + all_fps_time);
        Log.d(TAG, "All time for Feature Selection: " + all_feature_selection);

        Log.d(TAG, "Average time for extracted features opencv: " + all_feature_extraction_time / image_num);
        Log.d(TAG, "Average time for MLP: " + all_mlp_time / image_num);
        Log.d(TAG, "Average time for FPS: " + all_fps_time / fps_image_num);
        Log.d(TAG, "Average time for Feature Selection: " + all_feature_selection / image_num);
    }
}