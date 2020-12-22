package com.example.myapplication;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.MediaScannerConnection;
import android.os.Environment;
import android.util.Base64;

import com.google.gson.JsonArray;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLEncoder;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Map;

public class HttpUtils {

    public static String submitPostCommand(int command) throws MalformedURLException {
        /**
         * 发送POST请求到服务器并返回服务器信息
         * @param params 请求体内容
         * @param encode 编码格式
         * @return 服务器返回信息
         */
        String spec = "http://192.168.1.104:1122/command?id=" + command;
        URL url = new URL(spec);
        HttpURLConnection httpURLConnection = null;


        try{
            httpURLConnection = (HttpURLConnection)url.openConnection();
            httpURLConnection.setConnectTimeout(10000);
            httpURLConnection.setDoInput(true);
            httpURLConnection.setDoOutput(true);
            httpURLConnection.setRequestMethod("GET");


            int response = httpURLConnection.getResponseCode();
            System.out.println("uploadFiledata"+response);

            if (response == HttpURLConnection.HTTP_OK) {
                InputStream inputStream = httpURLConnection.getInputStream();
                String r = httpURLConnection.getResponseMessage();
                System.out.println("newoneresult : " + r);
                if(inputStream==null)
                    return "0";
                else
                    return "OK";
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            httpURLConnection.disconnect();
        }

        return "0";
    }

    public static String submitPostRoute(JsonArray routes) throws MalformedURLException {
        /**
         * 发送POST请求到服务器并返回服务器信息
         * @param params 请求体内容
         * @param encode 编码格式
         * @return 服务器返回信息
         */
        //byte[] data = getRequestData(params, encode).toString().getBytes();
        String spec = "http://192.168.1.104:1122/route";
        URL url = new URL(spec);
        HttpURLConnection httpURLConnection = null;


        try{
            httpURLConnection = (HttpURLConnection)url.openConnection();
            httpURLConnection.setConnectTimeout(10000);
            httpURLConnection.setDoInput(true);
            httpURLConnection.setDoOutput(true);
            httpURLConnection.setRequestMethod("POST");


            byte[] data = new byte[100];
            OutputStream out = httpURLConnection.getOutputStream();
            data = routes.toString().getBytes();
            out.write(data);
            out.flush();
            out.close();

            int response = httpURLConnection.getResponseCode();
            System.out.println("uploadFiledata"+response);

            if (response == HttpURLConnection.HTTP_OK) {
                InputStream inputStream = httpURLConnection.getInputStream();
                String r = httpURLConnection.getResponseMessage();
                System.out.println("newoneresult : " + r);
                if(inputStream==null)
                    return "0";
                else
                    return "OK";
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            httpURLConnection.disconnect();
        }
        return "0";
    }






}
