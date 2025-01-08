let serverUrl="http://localhost:8000"



function showMessage(type,msg){

    
    document.getElementById('message').classList.add("error");
    if(type=="error"){
        document.getElementById('message').style.color = 'red'; 
    }else{
        document.getElementById('message').style.color = 'green'; 
    }
  
    document.getElementById('message').innerText = type.charAt(0).toUpperCase() + type.slice(1) + ":" + msg 
    document.getElementById('message').style.display = 'block'; 
}

function hideMessage(){
    document.getElementById('message').style.display = 'none'; 
}



function managePage(action){
    console.log('Manage page .. action '+ action)

    const allBtn = document.querySelectorAll('button')
       allBtn.forEach((button) => {       
        console.log(button.id)
        if(action == "disable" && button.id != "toggle"){
            console.log('disable...')
            button.disabled = true;
        }
        else if(button.id != "toggle"){
            console.log('enable...')
            button.disabled = false;
        }


    });
}

function getSelectedPage() {
    const selectedRadio = document.querySelector('input[name="page"]:checked');
    return selectedRadio ? selectedRadio.value : 'ros'; // Default to 'ros' if nothing is selected
}

// Update your existing submit function in run_transcript.html to use getSelectedPage()
function submit() {
    // ... existing code ...
    const selectedPage = getSelectedPage();
    // ... rest of the submit function ...
}

// Update your runEvaluation function in run_eval.html to use getSelectedPage()
function runEvaluation() {
    // ... existing code ...
    const selectedPage = getSelectedPage();
    // Add the selectedPage to your API call body
    body: JSON.stringify({
        csv_data: base64Content,
        page: selectedPage
    })
    // ... rest of the function ...
}

  function loadComponents(){
        // Load components
        fetch('../common/html/sidebar.html').then(response => response.text()).then(data => {
            document.getElementById('sidebar').innerHTML = data;
            // Add this: Set active class on current page link
            const currentPath = window.location.pathname;
            const sidebarLinks = document.querySelectorAll('#sidebar a');
            // First remove any existing active classes
            sidebarLinks.forEach(link => link.classList.remove('active'));
            // Then add active class to current page link
            sidebarLinks.forEach(link => {
                if (link.getAttribute('href') === currentPath) {
                    link.classList.add('active');
                }
            });
        });
        fetch('../common/html/header.html').then(response => response.text()).then(data => document.getElementById('header').innerHTML = data);
        fetch('../common/html/page_selector.html').then(response => response.text()).then(data => document.getElementById('pageSelector').innerHTML = data);
    }
