package com.example.myapplication;

import android.graphics.Bitmap;
import android.os.Parcel;
import android.os.Parcelable;

import java.io.Serializable;


public class ImageItem implements Serializable, Parcelable {

    public String name;
    public String path;
    public long size;
    public int width;
    public int height;
    public String mimeType;
    public long addTime;

    @Override
    public boolean equals(Object o) {
        if (o instanceof ImageItem) {
            ImageItem item = (ImageItem) o;
            return this.path.equalsIgnoreCase(item.path) && this.addTime == item.addTime;
        }

        return super.equals(o);
    }

    public ImageItem(Bitmap bitmap, String path){
        this.name = bitmap.toString();
        this.path = path;
        this.size = bitmap.getAllocationByteCount();
        this.width = bitmap.getWidth();
        this.height = bitmap.getHeight();
        this.mimeType = "Bitmap";
        this.addTime = 5555555;
    }


    @Override
    public int describeContents() {
        return 0;
    }

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeString(this.name);
        dest.writeString(this.path);
        dest.writeLong(this.size);
        dest.writeInt(this.width);
        dest.writeInt(this.height);
        dest.writeString(this.mimeType);
        dest.writeLong(this.addTime);
    }

    public ImageItem() {
    }

    protected ImageItem(Parcel in) {
        this.name = in.readString();
        this.path = in.readString();
        this.size = in.readLong();
        this.width = in.readInt();
        this.height = in.readInt();
        this.mimeType = in.readString();
        this.addTime = in.readLong();
    }

    public static final Creator<ImageItem> CREATOR = new Creator<ImageItem>() {
        @Override
        public ImageItem createFromParcel(Parcel source) {
            return new ImageItem(source);
        }

        @Override
        public ImageItem[] newArray(int size) {
            return new ImageItem[size];
        }
    };
}
