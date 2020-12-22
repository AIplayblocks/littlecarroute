package com.example.myapplication;

import androidx.annotation.ColorRes;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PointF;
import android.graphics.RectF;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;

import org.json.JSONArray;

import java.net.MalformedURLException;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    private int LEFT = 1;
    private int RIGHT = 2;
    private int FORWARD = 3;
    private int BACKWARD = 4;
    private int routeFlag = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(new RouteView(this));


    }



    class RouteView extends View{
        Paint paint;
        private ArrayList<PointF> graphics = new ArrayList<PointF>();
        private ArrayList<PointF> routes = new ArrayList<PointF>();

        public RouteView(Context context) {
            super(context);
            paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStrokeJoin(Paint.Join.ROUND);
            paint.setStrokeCap(Paint.Cap.ROUND);
            paint.setStrokeWidth(10);
        }



        @Override
        public boolean onTouchEvent(MotionEvent event) {
            DisplayMetrics metric = new DisplayMetrics();
            getWindowManager().getDefaultDisplay().getMetrics(metric);
            int screenWidth = metric.widthPixels;
            int screenHeight = metric.heightPixels;


            int center = screenWidth/2;
            // Draw
            if(event.getX() > center - 400 && event.getX() < center + 400 && event.getY() > center - 400 && event.getY() < center + 400) {

                if(routeFlag == 1){
                    routes.add(new PointF(event.getX(), event.getY()));
                }
                else{
                    graphics.add(new PointF(event.getX(), event.getY()));
                }
                invalidate();
            }

            // Route Flag setting
            else if(event.getX() > center - 100 && event.getX() < center + 100 && event.getY() > screenHeight - 800 && event.getY() < screenHeight - 700){
                System.out.println(routeFlag);
                routeFlag = 1;
            }

            // Clear
            else if(event.getX() > center +150 && event.getX() < center + 350 && event.getY() > screenHeight - 800 && event.getY() < screenHeight - 700){
                graphics.clear();
                routes.clear();
                routeFlag = 0;

                invalidate();
            }

            // Confirm
            else if(event.getX() > center - 350 && event.getX() < center - 150 && event.getY() > screenHeight - 800 && event.getY() < screenHeight - 700){
                if(graphics.size() > 2){
                    graphics.add(new PointF(graphics.get(0).x, graphics.get(0).y));
                    invalidate();
                }


            }

            // Send
            else if(event.getX() > center - 100 && event.getX() < center + 100 && event.getY() > screenHeight - 600 && event.getY() < screenHeight - 500){
                new Thread(new Runnable(){
                    @Override
                    public void run() {
                        try {
                            Gson gson = new GsonBuilder().create();

                            JsonArray routesArray = gson.toJsonTree(routes).getAsJsonArray();
                            System.out.println(routesArray);
                            HttpUtils.submitPostRoute(routesArray);
                        } catch (MalformedURLException e) {
                            e.printStackTrace();
                        }
                    }
                }).start();
            }

            // Turn left
            else if(event.getX() > center - 550 && event.getX() < center - 400 && event.getY() > screenHeight - 500 && event.getY() < screenHeight - 400){
                new Thread(new Runnable(){
                    @Override
                    public void run() {
                        try {
                            HttpUtils.submitPostCommand(LEFT);
                        } catch (MalformedURLException e) {
                            e.printStackTrace();
                        }
                    }
                }).start();
            }
            // Turn Right
            else if(event.getX() > center - 350 && event.getX() < center - 200 && event.getY() > screenHeight - 500 && event.getY() < screenHeight - 400){
                new Thread(new Runnable(){
                    @Override
                    public void run() {
                        try {
                            HttpUtils.submitPostCommand(RIGHT);
                        } catch (MalformedURLException e) {
                            e.printStackTrace();
                        }
                    }
                }).start();
            }
            // Forward
            else if(event.getX() > center - 425 && event.getX() < center - 325 && event.getY() > screenHeight - 670 && event.getY() < screenHeight - 520){
                new Thread(new Runnable(){
                    @Override
                    public void run() {
                        try {
                            HttpUtils.submitPostCommand(FORWARD);
                        } catch (MalformedURLException e) {
                            e.printStackTrace();
                        }
                    }
                }).start();
            }
            // Backward
            else if(event.getX() > center - 425 && event.getX() < center - 325 && event.getY() > screenHeight - 380 && event.getY() < screenHeight - 230){
                new Thread(new Runnable(){
                    @Override
                    public void run() {
                        try {
                            HttpUtils.submitPostCommand(BACKWARD);
                        } catch (MalformedURLException e) {
                            e.printStackTrace();
                        }
                    }
                }).start();
            }

            return true;
        }

        private void drawController(int pixel,DisplayMetrics metric) {


        }
        @Override
        protected void onDraw(Canvas canvas) {
            DisplayMetrics metric = new DisplayMetrics();
            getWindowManager().getDefaultDisplay().getMetrics(metric);
            int screenWidth = metric.widthPixels;
            int screenHeight = metric.heightPixels;


            int center = screenWidth/2;
            System.out.println(center);
            System.out.println(screenHeight);

            canvas.drawColor(Color.GRAY);

            // Draw center canvas
            paint.setColor(Color.WHITE);
            canvas.drawRect(center - 400,center - 400,center + 400,center + 400,paint);

            // Draw cancel button
            canvas.drawRect(center + 150,screenHeight - 800,center + 350,screenHeight - 700,paint);
            // Draw metric
            paint.setColor(Color.GRAY);
            canvas.drawRect(center - 20,center - 20,center + 20,center + 20,paint);

            // Draw route button
            paint.setColor(Color.BLUE);
            canvas.drawRect(center - 100,screenHeight - 800,center + 100,screenHeight - 700,paint);

            // Draw send button
            paint.setColor(Color.CYAN);
            canvas.drawRect(center - 100,screenHeight - 600,center + 100,screenHeight - 500,paint);

            // Draw confirm button
            paint.setColor(Color.RED);
            canvas.drawRect(center - 350,screenHeight - 800,center - 150,screenHeight - 700,paint);

            paint.setColor(Color.GREEN);
            //Left Button
            canvas.drawRect(center - 550,screenHeight - 500,center - 400,screenHeight - 400,paint);
            //Right Button
            canvas.drawRect(center - 350,screenHeight - 500,center - 200,screenHeight - 400,paint);
            //Forward Button
            canvas.drawRect(center - 425,screenHeight - 670,center - 325,screenHeight - 520,paint);
            //Backward Button
            canvas.drawRect(center - 425,screenHeight - 380,center - 325,screenHeight - 230,paint);

            int cnt = 0;
            paint.setColor(Color.RED);
            for (PointF point : graphics) {
                    canvas.drawPoint(point.x, point.y, paint);
                    if (cnt > 0) {
                        canvas.drawLine(point.x, point.y, graphics.get(cnt - 1).x, graphics.get(cnt - 1).y, paint);
                    }
                    cnt++;
            }
            cnt = 0;
            paint.setColor(Color.BLUE);
            for (PointF point : routes) {
                canvas.drawPoint(point.x, point.y, paint);
                if (cnt > 0) {
                    canvas.drawLine(point.x, point.y, routes.get(cnt - 1).x, routes.get(cnt - 1).y, paint);
                }
                cnt++;
            }


        }
    }
}