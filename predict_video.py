from ai_video_detector import predict

video_path = "test_video.mp4"
model_path = "model.pth"

result = predict(video_path, model_path)

print("\n=== RESULT ===")

if result > 0.5:
    print("Prediction: Fake")
else:
    print("Prediction: Real")

print("Confidence:", result)
