document.addEventListener("DOMContentLoaded", async function () {
    let result_veracity = "";
    let result_credibility = "";
    let result_explanation = "";
    let result_impact_level = "";
    let currentTabUrl = "";
    let articleTitle = "";
    
    // Show loading state immediately
    updateUI("Loading...", "...", "Analyzing the current page...", "");

    chrome.tabs.query({ active: true, currentWindow: true }, async function (tabs) {
        if (!tabs || tabs.length === 0 || !tabs[0]) {
            console.error("No active tab found.");
            updateUI("Error", "0", "No active tab found.", "");
            return;
        }

        if (!tabs[0].url || tabs[0].url.startsWith("chrome://") || tabs[0].url.startsWith("chrome-extension://") || tabs[0].url.startsWith("file://")) {
            console.error("Invalid or restricted URL.");
            updateUI("Error", "0", "Invalid or restricted URL. Please navigate to a webpage.", "");
            return;
        }

        currentTabUrl = tabs[0].url;
        console.log("Current URL:", currentTabUrl);

        // Execute script in the current tab to get the page title
        chrome.scripting.executeScript({
            target: { tabId: tabs[0].id },
            function: getPageTitle
        }, async (results) => {
            if (chrome.runtime.lastError) {
                console.error("Error executing script:", chrome.runtime.lastError);
                updateUI("Error", "0", "Failed to extract page title.", "");
                return;
            }

            if (results && results[0] && results[0].result) {
                articleTitle = results[0].result;
                console.log("Article title:", articleTitle);
                
                try {
                    // Now that we have the title, send it to the API
                    const response_result = await verifyArticle(articleTitle);
                    console.log("Response result:", response_result);
                    
                    if (response_result) {
                        result_veracity = response_result.veracity ? "True" : "False";
                        result_credibility = Math.round(response_result.confidence_score * 100);
                        result_explanation = response_result.explanation || "No explanation available.";
                        result_impact_level = response_result.impact_level || "UNKNOWN";
                        
                        updateUI(result_veracity, result_credibility, result_explanation, result_impact_level);
                    }
                } catch (error) {
                    console.error("Error processing API response:", error);
                    updateUI("Error", "0", `Error processing API response: ${error.message}`, "");
                }
            } else {
                updateUI("Error", "0", "Failed to extract page title.", "");
            }
        });
    });

    let closeBtn = document.getElementById("close");
    if (closeBtn) {
        closeBtn.addEventListener("click", function () {
            window.close();
        });
    }
});

// Function to get the page title (executed in the context of the page)
function getPageTitle() {
    return document.title || "";
}

// Function to safely update the UI
function updateUI(veracity, credibility, explanation, impactLevel) {
    const veracityElem = document.getElementById("veracity");
    const credibilityElem = document.getElementById("credibility");
    const reasonElem = document.getElementById("reason");
    const impactLevelElem = document.getElementById("impact-level");

    if (veracityElem) veracityElem.innerText = `Veracity: ${veracity}`;
    if (credibilityElem) credibilityElem.innerText = `Credibility: ${credibility}%`;
    if (reasonElem) reasonElem.innerText = `${explanation}`;
    
    // Add the impact level to the UI
    if (impactLevelElem) {
        if (impactLevel) {
            impactLevelElem.innerText = `Impact Level: ${impactLevel}`;
            impactLevelElem.style.display = "block";
        } else {
            impactLevelElem.style.display = "none";
        }
    }
    
    // Hide loading indicator if it exists
    const loadingIndicator = document.getElementById("loading-indicator");
    if (loadingIndicator) {
        loadingIndicator.style.display = veracity === "Loading..." ? "flex" : "none";
    }
}

// Function to verify the article with the API
async function verifyArticle(title) {
    if (!title || title.trim() === "") {
        console.error("Error: No valid title provided.");
        return {
            veracity: false,
            confidence_score: 0,
            explanation: "No valid title provided. Please navigate to a webpage with a title and try again.",
            impact_level: ""
        };
    }

    try {
        const response = await fetch("http://127.0.0.1:8000/api/verify-claim/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ article_title: title })
        });

        if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error("Error verifying article:", error);
        return {
            veracity: false,
            confidence_score: 0,
            explanation: `Error processing request: ${error.message}`,
            impact_level: ""
        };
    }
}
