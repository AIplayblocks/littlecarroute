package com.example.myapplication;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PointF;
import android.graphics.RectF;
import android.graphics.Shader;
import android.view.MotionEvent;
import android.view.View;

import java.util.ArrayList;

public class DrawView extends View {
    public DrawView(Context context) {
        super(context);
    }
    private ArrayList<PointF> graphics = new ArrayList<PointF>();

    @Override
    public boolean onTouchEvent(MotionEvent event) {

        graphics.add(new PointF(event.getX(),event.getY()));
        invalidate();
        Canvas canvas = new Canvas();
        this.Draw(canvas);
        return true;
    }

    protected void Draw(Canvas canvas) {
        Paint p = new Paint();
        p.setColor(Color.RED);

        for (PointF point : graphics) {
            System.out.println(point.x);
            canvas.drawPoint(point.x, point.y, p);
        }


    }


}
