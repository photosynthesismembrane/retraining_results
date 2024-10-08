let currentIndex = 0;
const imagesPerPage = 20;
let selectedQuestions = [];
let image_folder = 'abstract_2500_generations_2';
let image_data = evaluation_abstract_data;

// Function to shuffle array elements
function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

function truncateText(text, maxChars) {
    if (text.length <= maxChars) {
        return text;
    }
    let truncated = text.substring(0, maxChars);
    const lastSentenceEnd = Math.max(truncated.lastIndexOf('.'), truncated.lastIndexOf('!'), truncated.lastIndexOf('?'));
    if (lastSentenceEnd > -1) {
        truncated = truncated.substring(0, lastSentenceEnd + 1);
    } else {
        truncated = truncated.substring(0, truncated.lastIndexOf(' '));
    }
    return truncated.trim();
}

// Function to populate questions
function populateQuestions() {
    const questionsContainer = document.getElementById('questions-container');
    questionsContainer.innerHTML = ``;
    questions.forEach(questionObj => {
        const questionItem = document.createElement('div');
        questionItem.className = 'question-item';
        questionItem.innerText = questionObj.question;
        questionItem.onclick = () => handleQuestionClick(questionItem, questionObj.label);
        questionsContainer.appendChild(questionItem);
    });
}

// Handle question click
function handleQuestionClick(questionItem, questionLabel) {

    const previouslySelected = document.querySelector('.question-item.selected');
    if (previouslySelected) {
        previouslySelected.classList.remove('selected');
    }
    questionItem.classList.add('selected');
    selectedQuestions = [questionLabel];

    loadImages(currentIndex);
}

function changeDataSource() {
    const dataSource = document.getElementById('data-source').value;
    if (dataSource === 'renaissance') {
        image_folder = 'renaissance_2500_generations_2';
        image_data = evaluation_renaissance_data;
    } else if (dataSource === 'landscape') {
        image_folder = 'landscape_2500_generations_2';
        image_data = evaluation_landscape_data;
    } else if (dataSource === 'abstract') {
        image_folder = 'abstract_2500_generations_2';
        image_data = evaluation_abstract_data;
    } 
    populateQuestions();
    loadImages(currentIndex);
}


// Load images
function loadImages(startIndex) {
    let model = 'llava';
    if (document.getElementById('llava').checked) {
        model = 'llava';
    } else if (document.getElementById('cogvlm').checked) {
        model = 'cogvlm';
    } else if (document.getElementById('deepseek').checked) {
        model = 'deepseek';
    }
    let iterations = '35k';
    if (document.getElementById('8k').checked) {
        iterations = '8k';
    } else if (document.getElementById('15k').checked) {
        iterations = '15k';
    } else if (document.getElementById('25k').checked) {
        iterations = '25k';
    } else if (document.getElementById('35k').checked) {
        iterations = '35k';
    }
    const mainPlot = document.getElementById('main-plot');   
    mainPlot.src = `clip_score_plots/${iterations}_diffusion_models_clip_score.png`;

    const dataSource = document.getElementById('data-source').value;
    const diffusion_models = ['stable_diffusion_v1_5', dataSource + '_llava_' + iterations, dataSource + '_cogvlm_' + iterations, dataSource + '_deepseek_' + iterations, dataSource + '_best_' + iterations];
    
    if (selectedQuestions.length === 0) {
        return;
    }

    const imageContainer = document.getElementById('image-container');
    imageContainer.innerHTML = ''; // Clear previous images

    const imagesArray = Object.entries(image_data).map(([key, value]) => ({
        image_filename: key,
        ...value
    })).filter(image => !noDataForImage(image)).slice(startIndex, startIndex + imagesPerPage);

    function imageModelAnswersEmpty(image, model) {
        if  (!`${model}_answers` in image) {
            return true;
        }

        // Check all selected questions whether there is at least one answer for the model
        return selectedQuestions.every(question => 
            !(image[`${model}_answers`] && question in image[`${model}_answers`] && image[`${model}_answers`][question])
        );
    }

    function noDataForImage(image) {
        const llavaChecked = document.getElementById('llava').checked;
        const cogvlmChecked = document.getElementById('cogvlm').checked;
        const deepseekChecked = document.getElementById('deepseek').checked;

        if ((llavaChecked && imageModelAnswersEmpty(image, 'llava')) && (cogvlmChecked && imageModelAnswersEmpty(image, 'cogvlm')) && (deepseekChecked && imageModelAnswersEmpty(image, 'deepseek'))) {
            return true;
        }
        return false;
    }

    // CHECK IF IMAGE EXISTS
    function checkIfImageExists(url, callback) {
        const img = new Image();
        img.src = url;
        
        if (img.complete) {
            callback(true);
        } else {
            img.onload = () => {
                callback(true);
            };
        
            img.onerror = () => {
                callback(false);
            };
        }
    }

    imagesArray.forEach((image, index) => {
        if (noDataForImage(image)) {
            return;
        }

        const answersDiv = document.createElement('div');
        answersDiv.className = 'answers';

        const llavaChecked = document.getElementById('llava').checked;
        const cogvlmChecked = document.getElementById('cogvlm').checked;
        const deepseekChecked = document.getElementById('deepseek').checked;

        if (llavaChecked && !imageModelAnswersEmpty(image, 'llava')) {
            const llavaDiv = document.createElement('div');
            llavaDiv.className = 'answer-box';
            text =  truncateText(selectedQuestions.map(question => image.llava_answers[question] || "N/A").join('<br>'), 1000);
            llavaDiv.innerHTML += `<p><strong>llava:</strong><br>${text}</p>`;
            answersDiv.appendChild(llavaDiv);
        }
        if (cogvlmChecked && !imageModelAnswersEmpty(image, 'cogvlm')) {
            const cogvlmDiv = document.createElement('div');
            cogvlmDiv.className = 'answer-box';
            text = truncateText(selectedQuestions.map(question => image.cogvlm_answers[question] || "N/A").join('<br>'), 1000);
            cogvlmDiv.innerHTML += `<p><strong>cogvlm:</strong><br>${text}</p>`;
            answersDiv.appendChild(cogvlmDiv);
        }
        if (deepseekChecked && !imageModelAnswersEmpty(image, 'deepseek')) {
            const deepseekDiv = document.createElement('div');
            deepseekDiv.className = 'answer-box';
            text = truncateText(selectedQuestions.map(question => image.deepseek_answers[question] || "N/A").join('<br>'), 1000);
            deepseekDiv.innerHTML += `<p><strong>deepseek:</strong><br>${text}</p>`;
            answersDiv.appendChild(deepseekDiv);
        }        

        const imageWrapper = document.createElement('div');
        imageWrapper.className = 'image-wrapper';

        imageWrapper.appendChild(answersDiv);

        for (let i = 0; i < diffusion_models.length; i++) {
            const diffusion_model = diffusion_models[i];
            const imgContainer = document.createElement('div');
            imgContainer.className = 'image-container';
            const imgElement = document.createElement('img');
            imgElement.className = 'image';
            imgElement.src = `${image_folder}/${diffusion_model}_${model}_${selectedQuestions[0]}_${image.image_filename.replace('.jpg', '')}.png`;
            imgElement.alt = imgElement.src;
            // imgElement.onclick = () => openModal(imgElement.src);
            imgContainer.appendChild(imgElement);
            imageWrapper.appendChild(imgContainer);
        }
        
        imageContainer.appendChild(imageWrapper);
    });
}


function openMainTab(tabId) {
    var i, tabContent, tabButtons;

    tabContent = document.getElementsByClassName("main-tab-content");
    for (i = 0; i < tabContent.length; i++) {
        tabContent[i].style.display = "none";
    }

    tabButtons = document.getElementsByClassName("main-tab-button");
    for (i = 0; i < tabButtons.length; i++) {
        tabButtons[i].classList.remove("active");
    }

    document.getElementById(tabId).style.display = "block";
    event.currentTarget.classList.add("active");
}

// Navigation functions
function loadPrevious() {
    if (currentIndex > 0) {
        currentIndex -= imagesPerPage;
        loadImages(currentIndex);
        scrollToFirstImage();
    }
}

function loadNext() {
    if (currentIndex + imagesPerPage < Object.keys(image_data).length) {
        currentIndex += imagesPerPage;
        loadImages(currentIndex);
        scrollToFirstImage();
    }
}

// Function to scroll to the first image
function scrollToFirstImage() {
    const firstImage = document.querySelector('#image-container img');
    if (firstImage) {
        firstImage.scrollIntoView({ behavior: 'smooth' });
    }
}


function openModal(src) {
    document.getElementById("myModal").style.display = "block";
    document.getElementById("modalImage").src = src;
}

function closeModal() {
    document.getElementById("myModal").style.display = "none";
}

// Close the modal when the user clicks anywhere outside of the modal content
window.onclick = function(event) {
    if (event.target == document.getElementById("myModal")) {
        closeModal();
    }
}


// Initialize
populateQuestions();

document.getElementById('llava').addEventListener('change', () => loadImages(currentIndex));
document.getElementById('cogvlm').addEventListener('change', () => loadImages(currentIndex));
document.getElementById('deepseek').addEventListener('change', () => loadImages(currentIndex));

document.getElementById('8k').addEventListener('change', () => loadImages(currentIndex));
document.getElementById('15k').addEventListener('change', () => loadImages(currentIndex));
document.getElementById('25k').addEventListener('change', () => loadImages(currentIndex));
document.getElementById('35k').addEventListener('change', () => loadImages(currentIndex));
document.getElementById('data-source').addEventListener('change', changeDataSource);

loadImages(currentIndex);

