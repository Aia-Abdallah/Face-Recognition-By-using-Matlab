
%%Testing Model:

clc;close;clear

%%testing modle for collaction data
c=webcam;
load myNet1;
faceDetector=vision.CascadeObjectDetector;
while true
    e=c.snapshot;
    bboxes =step(faceDetector,e);
    if(sum(sum(bboxes))~=0)
     es=imcrop(e,bboxes(1,:));
    es=imresize(es,[227 227]);
    label=classify(myNet1,es);
    image(e);
    title(char(label));
    drawnow;
    else
        image(e);
        title('No Face Detected');
    end
end







% Initialize webcam object
c = webcam;

% Load the pre-trained face recognition network
load myNet1;  % Assumes 'myNet1.mat' is in your workspace

% Create a cascade object detector for face recognition
faceDetector = vision.CascadeObjectDetector;  % Pre-trained model likely needed

while true  % Infinite loop (can be modified to run for a specific duration)
    % Capture a frame from the webcam
    e = c.snapshot;

    % Detect faces in the captured frame
    bboxes = step(faceDetector, e);

    % Check if any faces were detected
    if (sum(sum(bboxes)) ~= 0)
        % If a face is detected:
        % Crop the captured frame based on the first bounding box
        es = imcrop(e, bboxes(1, :));  % Assuming you only want the first face

        % Resize the cropped image to the expected size for the network
        es = imresize(es, [227 227]);  % Might need adjustment based on your network

        % Classify the resized image using the loaded network
        label = classify(myNet1, es);

        % Display the original captured frame
        image(e);

        % Display the classification label as the title of the image
        title(char(label));

        % Update the graphics window to show the changes
        drawnow;
    else
        % If no face is detected:
        % Display the original captured frame
        image(e);

        % Display a message indicating no face detection
        title('No Face Detected');

        % Update the graphics window
        drawnow;
    end
end

% The loop continues until manually stopped (e.g., using keyboard interrupt)
