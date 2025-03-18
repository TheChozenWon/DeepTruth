import React, { useState, useEffect } from "react";
import ReactDOM from "react-dom";
import { Player } from "@lottiefiles/react-lottie-player";
import animationData from "./loading.json";
import "./Nav.css";
import "./App.css";
import axios from "axios";

function App() {
  const [searchVisible, setSearchVisible] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [loading, setLoading] = useState(false);
  const [credibility, setCredibility] = useState(0);
  const [authenticity, setAuthenticity] = useState(null);
  const [explanation, setExplanation] = useState("");
  const [sources, setSources] = useState([]);
  const [goClicked, setGoClicked] = useState(false);
  const [animatedCredibility, setAnimatedCredibility] = useState(0);

  const handleSearchClick = () => {
    setSearchVisible(true);
  };

  const handleGoClick = () => {
    setLoading(true);
    setGoClicked(true);
    
    axios.post('http://127.0.0.1:8000/api/verify-claim/', { article_title: searchText })
      .then(response => {
        setLoading(false);
        const data = response.data;
        setCredibility(data.confidence_score * 100);
        setAuthenticity(data.veracity);
        setExplanation(data.explanation);
        
        // Convert the sources string to an array and format for display
        let sourcesArray = [];
        try {
          // Remove the single quotes and square brackets from the string
          const sourcesString = data.sources.replace(/^\['|'\]$/g, '').replace(/'\s*,\s*'/g, ',');
          sourcesArray = sourcesString.split(',').map((url, index) => ({
            url: url,
            title: `Source ${index + 1}`
          }));
        } catch (error) {
          console.error("Error parsing sources:", error);
          sourcesArray = [];
        }
        
        setSources(sourcesArray);
      })
      .catch(error => {
        console.error('Error:', error);
        setLoading(false);
        setCredibility(0);
        setAuthenticity(null);
        setExplanation("An error occurred while verifying the claim.");
        setSources([]);
      });
  };

  const handleResetClick = () => {
    setSearchVisible(false);
    setSearchText("");
    setLoading(false);
    setCredibility(0);
    setAuthenticity(null);
    setExplanation("");
    setSources([]);
    setGoClicked(false);
  };

  useEffect(() => {
    if (goClicked) {
      const interval = setInterval(() => {
        setAnimatedCredibility((prev) => {
          if (prev < credibility) {

            return Math.min(prev + 1, credibility);
          }
          clearInterval(interval);
          return prev;
        });
      }, 20);

      return () => clearInterval(interval);
    }
  }, [credibility, goClicked]);

  const renderCircularProgress = (percentage, size = 150, thickness = 10) => {
    const normalizedPercentage = Math.min(100, Math.max(0, percentage));
    const radius = (size - thickness) / 2;
    const circumference = 2 * Math.PI * radius;
    const strokeDashoffset = circumference - (normalizedPercentage / 100) * circumference;

    return (
      <div className="neon-circular-progress" style={{ display: "flex", alignItems: "center" }}>
        <div style={{ position: "relative" }}>
          <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
            <circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              fill="transparent"
              stroke="rgba(255, 255, 255, 0.1)"
              strokeWidth={thickness}
            />
            <circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              fill="transparent"
              stroke="#b700ff"
              strokeWidth={thickness}
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              className="neon-glow-circle"
            />
          </svg>
          <div
            className="neon-percentage"
            style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              fontSize: "1.5em",
              color: "#b700ff",
            }}
          >
            {Math.round(normalizedPercentage)}%
          </div>
        </div>
      </div>
    );
  };

  return (
    <React.Fragment>
      <h1>News Credibility Checker</h1>

      {!goClicked && (
        <React.Fragment>
          <button onClick={handleSearchClick}>Search</button>

          {searchVisible && (
            <React.Fragment>
              <input
                type="text"
                placeholder="Enter News Article Title"
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
              />
              <button onClick={handleGoClick}>Go</button>
            </React.Fragment>
          )}
        </React.Fragment>
      )}
      {goClicked && <button onClick={handleResetClick}>Reset</button>}

      {loading && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginTop: "20px",
          }}
        >
          <Player
            autoplay
            loop
            src={animationData}
            style={{ width: "50px", height: "50px", marginRight: "10px" }}
          />
          <p>Loading...</p>
        </div>
      )}
      {!loading && credibility !== 0 && (
        <React.Fragment>
          <div id="load">
            <div
              className="credibility-container"
              style={{ flex: 1, textAlign: "center", fontSize: "1.5em" }}
            >
              <h2
                style={{
                  color: "#ffffff",
                  fontSize: "1.5em",
                  fontWeight: "bold",
                }}
              >
                Credibility
              </h2>
              {renderCircularProgress(animatedCredibility, 180)}
            </div>
            <div
              className="authenticity-container"
              style={{ flex: 1, textAlign: "center" }}
            >
              <p style={{ color: "#ffffff", fontSize: "2em" }}>
                Veracity: {authenticity ? "True" : "False"}
              </p>
            </div>
          </div>

          <div id="explanation">
            <p
              style={{
                maxWidth: "800px",
                textAlign: "center",
                color: "#ffffff",
              }}
            >
              {explanation}
            </p>
          </div>

          <h3
            style={{
              textAlign: "center",
              color: "#ffffff",
            }}
          >
            Sources:
          </h3>
          <ul>
            {sources.map((source, index) => (
              <li
                key={index}
                style={{
                  flex: 1,
                  textAlign: "center",
                  borderStyle: "solid",
                  borderWidth: "1px",
                  padding: "5px",
                }}
              >
                <a href={source.url} style={{ color: "#00ffcc" }}>
                  {source.title}
                </a>
              </li>
            ))}
          </ul>
        </React.Fragment>
      )}
    </React.Fragment>
  );
}

ReactDOM.render(<App />, document.getElementById("root"));

export default App;
