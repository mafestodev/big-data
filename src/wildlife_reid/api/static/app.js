document.addEventListener("DOMContentLoaded", () => {
    const dashboardScreen = document.getElementById("dashboard-screen");
    const uploadScreen = document.getElementById("upload-screen");

    const identifyAnimal = document.getElementById("identify-animal");
    const identifyNav = document.getElementById("identify-nav");
    const dashboardNav = document.getElementById("dashboard-nav");
    const backDashboard = document.getElementById("back-dashboard");
    const brand = document.querySelector(".brand");

    const imageInput = document.getElementById("image-input");
    const browseButton = document.getElementById("browse-button");
    const emptyUploadState = document.getElementById("empty-upload-state");
    const uploadedState = document.getElementById("uploaded-state");
    const uploadedPreview = document.getElementById("uploaded-preview");
    const analyseButton = document.getElementById("analyse-button");
    const removeImage = document.getElementById("remove-image");

    let selectedFile = null;

    //switch between dashboard and upload page
   function openUploadPage() {
    dashboardScreen.classList.add("hidden");
    uploadScreen.classList.remove("hidden");

    dashboardNav.classList.remove("active");
    identifyNav.classList.add("active");

    window.scrollTo(0, 0);
}

function openDashboard() {
    uploadScreen.classList.add("hidden");

    const resultScreen = document.getElementById("result-screen");

    resultScreen?.classList.add("hidden");

    dashboardScreen.classList.remove("hidden");

    identifyNav.classList.remove("active");
    dashboardNav.classList.add("active");

    window.scrollTo(0, 0);
}

    identifyAnimal?.addEventListener("click", openUploadPage);
    identifyNav?.addEventListener("click", openUploadPage);
    dashboardNav?.addEventListener("click", openDashboard)
    backDashboard?.addEventListener("click", openDashboard);
    brand?.addEventListener("click", openDashboard);

    //open file picker
    browseButton?.addEventListener("click", (event) => {
        event.stopPropagation();
        imageInput.click();
    });

    emptyUploadState?.addEventListener("click", (event) => {
        if (!event.target.closest("button")) {
            imageInput.click();
        }
    });

    // show uploaded image
    function showImage(file) {
        if (!file) return;

        if (!file.type.startsWith("image/")) {
            alert("Please choose an image file.");
            return;
        }

        selectedFile = file;

        const reader = new FileReader();

        reader.onload = (event) => {
            uploadedPreview.src = event.target.result;

            emptyUploadState.classList.add("hidden");

            uploadedState.classList.remove("hidden");
            analyseButton.classList.remove("hidden");
        };

        reader.readAsDataURL(file);
    }

    imageInput?.addEventListener("change", () => {
        const file = imageInput.files[0];
        showImage(file);
    });

    // drag and drop
    emptyUploadState?.addEventListener("dragover", (event) => {
        event.preventDefault();
        emptyUploadState.style.borderColor = "var(--forest-mid)";
    });

    emptyUploadState?.addEventListener("dragleave", () => {
        emptyUploadState.style.borderColor = "";
    });

    emptyUploadState?.addEventListener("drop", (event) => {
        event.preventDefault();
        emptyUploadState.style.borderColor = "";

        const file = event.dataTransfer.files[0];
        showImage(file);
    });

    // remove current image
    removeImage?.addEventListener("click", () => {
        selectedFile = null;

        imageInput.value = "";
        uploadedPreview.src = "";

        uploadedState.classList.add("hidden");
        analyseButton.classList.add("hidden");

        emptyUploadState.classList.remove("hidden");
    });

//Processing uploadedd image
const processingPanel = document.getElementById("processing-panel");
const processingSteps = document.querySelectorAll(".processing-step");

//Image results 
const resultScreen = document.getElementById("result-screen");
const resultQueryImage = document.getElementById("result-query-image");
const resultMatchImage = document.getElementById("result-match-image");
const resultBack = document.getElementById("result-back");
const identifyAgain = document.getElementById("identify-again");
analyseButton?.addEventListener("click", async () => {
    if (!selectedFile) {
        alert("Please upload an image first.");
        return;
    }

    analyseButton.classList.add("hidden");
    processingPanel.classList.remove("hidden");

    processingSteps.forEach((item, index) => {
        item.classList.remove("active", "done");
        item.querySelector(".processing-circle").textContent = index + 1;
    });

    const apiRequest = identifyWithApi();

    for (let i = 0; i < processingSteps.length; i++) {
        processingSteps[i].classList.add("active");

        await new Promise(resolve => setTimeout(resolve, 700));

        processingSteps[i].classList.remove("active");
        processingSteps[i].classList.add("done");
        processingSteps[i].querySelector(".processing-circle").textContent = "✓";
    }

    try {
        const data = await apiRequest;

        console.log("API response:", data);

        resultQueryImage.src = uploadedPreview.src;

        uploadScreen.classList.add("hidden");
        resultScreen.classList.remove("hidden");

        window.scrollTo(0, 0);
    } catch (error) {
        alert(error.message);

        processingPanel.classList.add("hidden");
        analyseButton.classList.remove("hidden");
    }
});

function goBackToUpload() {
    resultScreen.classList.add("hidden");
    uploadScreen.classList.remove("hidden");

    processingPanel.classList.add("hidden");
    analyseButton.classList.remove("hidden");

    processingSteps.forEach((item, index) => {
        item.classList.remove("active", "done");
        item.querySelector(".processing-circle").textContent = index + 1;
    });

    window.scrollTo(0, 0);
}

resultBack?.addEventListener("click", goBackToUpload);
identifyAgain?.addEventListener("click", goBackToUpload);

//Add the API call
const resultIdentity = document.getElementById("result-identity");
const resultSpecies = document.getElementById("result-species");
const resultSimilarity = document.getElementById("result-similarity");
const resultExplanation = document.getElementById("result-explanation-text");

async function identifyWithApi() {
    if (!selectedFile) {
        throw new Error("Please upload an image first.");
    }

    const formData = new FormData();
    formData.append("image", selectedFile);

    const response = await fetch("/api/v1/identify", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

  if (response.status === 503) {
    throw new Error(
        "The API is connected, but the trained model and gallery are not loaded yet."
    );
}

if (!response.ok) {
    throw new Error(data.error || "Unable to identify image.");
}

    resultIdentity.textContent = data.predicted_identity;
    resultSpecies.textContent = data.species;
    resultExplanation.textContent =
        `The query embedding is most similar to the stored reference ` +
        `embedding for individual ${data.predicted_identity}.`;
    resultMatchImage.src = data.matched_image_url;
    resultMatchImage.alt = `Best match for ${data.predicted_identity}`;

    const similarityPercent = data.similarity_score * 100;
    resultSimilarity.textContent = `${similarityPercent.toFixed(1)}%`;

    return data;
}

});
