/**
 * Return a request object
 */
 function getRequestObject() {
    if (window.XMLHttpRequest) {
        return (new XMLHttpRequest());
    } 
    else {
        window.alert("Ajax is not supported!");
        return(null); 
    }
}

/**
 * Wrapper for handling the response of a request
 * 
 * @param {request object} request 
 * @param {function for handling the response of the request} responseHandler 
 */
function handleResponse(request, responseHandler) {
    console.log(`Ready State ${request.readyState}`);
    console.log(`status ${request.status}`);
    if ((request.readyState === 4) && (request.status === 200)) {
        responseHandler(request);
    }
}
  
/**
 * Make request for a given URL with a given method
 * 
 * @param {url for the request} url 
 * @param {handler for the response} responseHandler 
 * @param {GET / POST / PUT / etc.} method 
 * @param {data to send if needed} data 
 */
function MakeRequest(url, responseHandler, method="GET", data=null, headers={}, async=true) {
    let requestor = getRequestObject();
    requestor.onreadystatechange = function() {
        handleResponse(requestor, responseHandler); 
    };
    requestor.open(method, url, async);
    if(headers) {
        for(var key in headers) {
            requestor.setRequestHeader(key, headers[key]);
        }
    }
    requestor.send(data);
    return requestor;
}

window.addEventListener("load", () => {
    let taskNameDom = document.getElementById("taskName");
    if (taskNameDom) {
        let taskName = taskNameDom.textContent;
        setInterval(() => {
            MakeRequest(
                `http://127.0.0.1:5000/TaskStatus/${taskName}`,
                (request) => {
                    let running = request.responseText;
                    if (running !== "True") {
                        console.log("Done!");
                        window.location.replace(`http://127.0.0.1:5000/TaskResult/${taskName}`);
                    }
                }
            );
        }, 2000);
    }
})
