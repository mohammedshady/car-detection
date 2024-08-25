import React, { useState, useEffect } from "react";
import axios from "axios";
import "./FileUpload.css";
import { Progress } from "antd";
import io from "socket.io-client";

function FileUpload() {
  const [file, setFile] = useState(null);
  const [apiStatus, setApiStatus] = useState([]);
  const [currentFile, setCurrentFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null); // State to hold the preview URL
  const [imageUrl, setImageUrl] = useState(null);
  const [progress, setProgress] = useState(0);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const socket = io.connect("http://localhost:5000");
    socket.on("progress_update", (data) => {
      const progressString = data.message;
      const progressIndex = progressString.indexOf("(");
      if (progressIndex !== -1) {
        const progressPercentage = parseFloat(
          progressString.substring(
            progressIndex + 1,
            progressString.indexOf("%")
          )
        );
        if (!isNaN(progressPercentage)) {
          setProgress(progressPercentage);

          const progressMessageWithoutPercentage = progressString.replace(
            /\(\d+\.\d+%\)/,
            ""
          );

          if (
            progressPercentage === 0 ||
            progressPercentage === 20 ||
            progressPercentage === 40 ||
            progressPercentage === 60 ||
            progressPercentage === 80 ||
            progressPercentage === 100
          ) {
            setApiStatus((prevApiStatus) => [
              ...prevApiStatus,
              progressMessageWithoutPercentage,
            ]);
          }
        }
      }
    });
    return () => {
      socket.disconnect();
    };
  }, [setApiStatus]);

  const onFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith("image")) {
      setFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        // Set the preview URL to the reader result
        setPreviewUrl(reader.result);
      };
      reader.readAsDataURL(file);
    } else {
      setFile(null);
      setPreviewUrl(null);
    }
  };
  const detectButtonHandler = async () => {
    try {
      setLoading(true);
      const response = await fetch(
        `http://localhost:5000/process/${currentFile}`
      );

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setImageUrl(url);
      console.log("Image processed successfully:", url);
    } catch (error) {
      console.error("Failed to fetch image:", error);
    } finally {
      setLoading(false);
    }
  };
  const onFileUpload = async () => {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/upload",
        formData
      );
      setCurrentFile(response.data.filename);
      console.log("File uploaded successfully:", response.data.filename);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div className="main-app-page">
      <div className="input-upload-image">
        <input type="file" onChange={onFileChange} />
        <button onClick={onFileUpload} className="button-file">
          Upload!
        </button>
      </div>
      {previewUrl && (
        <>
          <div className="image-input-output-contatiner">
            <div className="input-output-area">
              <h3>Original Image</h3>
              <img
                src={previewUrl}
                alt="Preview"
                style={{ width: "100%", height: "auto" }}
              />
            </div>

            <div className="input-output-area">
              <h3>Detection Boxes:</h3>
              {imageUrl && (
                <img
                  src={imageUrl}
                  alt="Preview"
                  style={{ width: "100%", height: "auto" }}
                />
              )}
            </div>
          </div>
          {!currentFile && <h4>Image needs to be uploaded first</h4>}
          <button
            disabled={loading || !currentFile}
            onClick={detectButtonHandler}
          >
            Detect !
          </button>
          {loading && (
            <div className="status-section-container">
              <h4 className="status-loading-title">
                {apiStatus[apiStatus.length - 1]}
              </h4>
              <Progress
                percent={progress}
                status="active"
                strokeColor={{ from: "#108ee9", to: "#87d068" }}
                style={{ color: "white", maxWidth: 400 }}
              />
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default FileUpload;
