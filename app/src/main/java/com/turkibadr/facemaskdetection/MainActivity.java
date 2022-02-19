package com.turkibadr.facemaskdetection;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.MediaPlayer;
import android.media.ThumbnailUtils;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.turkibadr.facemaskdetection.ml.Model;
import com.turkibadr.facemaskdetection.ml.ModelUnquantA;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    Button openCamera;
    ImageView imageView;
    int imageSize = 224;
    TextView txtView , result;
    MediaPlayer right , wrong;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        openCamera = findViewById(R.id.openCameraBtn);
        imageView = findViewById(R.id.imageView);
        txtView = findViewById(R.id.txtView);
        result = findViewById(R.id.result);
        right =  MediaPlayer.create(this,R.raw.right);
        wrong =  MediaPlayer.create(this,R.raw.wrong);


        openCamera.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    //Request camera permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });



    }// end onCreate

    private void classifyImage (Bitmap image){

        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect( 4 * imageSize * imageSize *3);
            byteBuffer.order(ByteOrder.nativeOrder());
            int[] intValues = new int[ imageSize * imageSize];
            image.getPixels(intValues , 0 , image.getWidth() , 0 , 0, image.getWidth() , image.getHeight());
            int pixel =0;
            for(int i = 0; i < imageSize ; i++){
                for(int j = 0 ; j < imageSize ; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF)  * (1.f / 255.f));

                }// end fot j
            }// end for i
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float [] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidences = 0;
            for(int i = 0; i < confidences.length; i++){
                if(confidences[i] > maxConfidences){
                    maxConfidences = confidences[i];
                    maxPos = i;
                }// end if
            }// end for
            String classes[] = {"wear Mask" , "do not wear mask"};
            //txtView.setText(classes[maxPos]);
            if(classes[maxPos]=="wear Mask"){
                right.start();
                Intent i = new Intent(this,WearingMask.class);
                startActivity(i);
            }else {
                wrong.start();
                Intent i = new Intent(this,DoNotWearingMask.class);
                startActivity(i);
            }

            String s = "";
            for(int t = 0; t < classes.length; t++){
                s+= String.format("%s : %.1f%%\n" , classes[t] , confidences[t] *100);
            }
           // result.setText(s);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }

    }// end classifyImage

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == 1 && resultCode == RESULT_OK) {
            Bitmap image = (Bitmap) data.getExtras().get("data");
            int dimension = Math.min(image.getWidth() , image.getHeight());
            image = ThumbnailUtils.extractThumbnail(image, dimension , dimension);
            imageView.setImageBitmap(image);

            image = Bitmap.createScaledBitmap(image, imageSize , imageSize , false);
            classifyImage(image);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }// end onActivityResult

}// end class


//try {
//        ModelUnquantA model = ModelUnquantA.newInstance(getApplicationContext());
//
//        // Creates inputs for reference.
//        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
//        ByteBuffer byteBuffer = ByteBuffer.allocateDirect( 4 * imageSize * imageSize *3);
//        byteBuffer.order(ByteOrder.nativeOrder());
//        int[] intValues = new int[ imageSize * imageSize];
//        image.getPixels(intValues , 0 , image.getWidth() , 0 , 0, image.getWidth() , image.getHeight());
//        int pixel =0;
//        for(int i = 0; i < imageSize ; i++){
//        for(int j = 0 ; j < imageSize ; j++){
//        int val = intValues[pixel++]; // RGB
//        byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
//        byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
//        byteBuffer.putFloat((val & 0xFF)  * (1.f / 255.f));
//
//        }// end fot j
//        }// end for i
//        inputFeature0.loadBuffer(byteBuffer);
//
//        // Runs model inference and gets result.
//        ModelUnquantA.Outputs outputs = model.process(inputFeature0);
//        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
//
//        float [] confidences = outputFeature0.getFloatArray();
//        int maxPos = 0;
//        float maxConfidences = 0;
//        for(int i = 0; i < confidences.length; i++){
//        if(confidences[i] > maxConfidences){
//        maxConfidences = confidences[i];
//        maxPos = i;
//        }// end if
//        }// end for
//        String classes[] = {"wear Mask" , "do not wear mask"};
//        txtView.setText(classes[maxPos]);
//
//        String s = "";
//        for(int t = 0; t < classes.length; t++){
//        s+= String.format("%s: %.1f%%\n" , classes[t] , confidences[t] *100);
//        }
//        result.setText(s);
//
//
//        // Releases model resources if no longer used.
//        model.close();
//        } catch (IOException e) {
//        // TODO Handle the exception
//        }

