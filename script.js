const container = document.getElementById("container");
const fileInput = document.getElementById("file-input");
const buttonStart = document.getElementById("start-btn");

let faceMatcher;

buttonStart.addEventListener("click", (e) => {
  const init = async () => {
    await Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
      faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
      faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
    ]);

    console.log("done model");

    const trainingData = await loadTrainingData();
    faceMatcher = new faceapi.FaceMatcher(trainingData, 0.6);
  };

  const loadTrainingData = async () => {
    const labels = ["putin", "trump"];

    const faceDescriptors = [];

    for (const label of labels) {
      const descriptors = [];
      for (let i = 1; i <= 34; i++) {
        const image = await faceapi.fetchImage(`/data/${label}/${i}.jpg`);
        const detection = await faceapi
          .detectSingleFace(image)
          .withFaceLandmarks()
          .withFaceDescriptor();
        console.log("det", detection);
        if (detection !== undefined) descriptors.push(detection.descriptor);
      }
      faceDescriptors.push(
        new faceapi.LabeledFaceDescriptors(label, descriptors)
      );
      console.log(label);
    }
    return faceDescriptors;
  };
  init();
});

fileInput.addEventListener("change", async (e) => {
  const file = fileInput.files[0];

  const image = await faceapi.bufferToImage(file);
  const canvas = faceapi.createCanvasFromMedia(image);
  container.innerHTML = "";
  container.append(image);
  container.append(canvas);

  const size = {
    width: image.width,
    height: image.height,
  };
  faceapi.matchDimensions(canvas, size);

  // Detect and art canvas
  const detections = await faceapi
    .detectAllFaces(image)
    .withFaceLandmarks()
    .withFaceDescriptors();
  const resizedDetections = faceapi.resizeResults(detections, size);

  console.log("1", faceMatcher);

  for (const detection of resizedDetections) {
    const box = detection.detection.box;
    const drawBox = new faceapi.draw.DrawBox(box, {
      label: faceMatcher.findBestMatch(detection.descriptor),
    });
    drawBox.draw(canvas);
  }
});
