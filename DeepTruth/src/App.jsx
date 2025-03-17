import React, { useState, useEffect } from "react";
import ReactDOM from "react-dom";
import { Player } from "@lottiefiles/react-lottie-player";
import animationData from "./loading.json"; // Path to your animation file
import "./Nav.css";
import "./App.css";

function App() {
  const [searchVisible, setSearchVisible] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [loading, setLoading] = useState(false);
  const [credibility, setCredibility] = useState(0); // Start with 0
  const [authenticity, setAuthenticity] = useState(null);
  const [explanation, setExplanation] = useState("");
  const [sources, setSources] = useState([]);
  const [goClicked, setGoClicked] = useState(false); // Track if Go button is clicked
  const [animatedCredibility, setAnimatedCredibility] = useState(0); // Animated credibility percentage

  const handleSearchClick = () => {
    setSearchVisible(true);
  };

  const handleGoClick = () => {
    setLoading(true);
    setGoClicked(true); // Set goClicked to true after clicking Go
    setTimeout(() => {
      setLoading(false);
      setCredibility(75); // Set credibility to 75 after loading
      setAuthenticity(true);
      setExplanation(
        "The young man wanted a role model. He looked long and hard in his youth, but that role model never materialized. His only choice was to embrace all the people in his life he didn't want to be like."
      );
      setSources([
        { url: "https://example.com/source1", title: "Source 1" },
        { url: "https://example.com/source2", title: "Source 2" },
        { url: "https://example.com/source3", title: "Source 3" },
        { url: "https://example.com/source4", title: "Source 4" },
        { url: "https://example.com/source5", title: "Source 5" },
      ]);
    }, 2000); // Simulate loading time
  };

  const handleResetClick = () => {
    setSearchVisible(false);
    setSearchText("");
    setLoading(false);
    setCredibility(0);
    setAuthenticity(null);
    setExplanation("");
    setSources([]);
    setGoClicked(false); // Reset goClicked state to false
  };

  // Animate the credibility percentage
  useEffect(() => {
    if (goClicked) {
      const interval = setInterval(() => {
        setAnimatedCredibility((prev) => {
          if (prev < credibility) {
            return Math.min(prev + 1, credibility); // Increase the percentage
          }
          clearInterval(interval);
          return prev; // Stop the animation when the target is reached
        });
      }, 20); // Adjust the speed of the animation (20ms interval)

      return () => clearInterval(interval); // Clean up the interval on unmount or re-render
    }
  }, [credibility, goClicked]);

  // Render circular progress bar
  const renderCircularProgress = (percentage, size = 150, thickness = 10) => {
    const normalizedPercentage = Math.min(100, Math.max(0, percentage));

    const radius = (size - thickness) / 2;
    const circumference = 2 * Math.PI * radius;
    const strokeDashoffset =
      circumference - (normalizedPercentage / 100) * circumference;

    return (
      <div
        className="neon-circular-progress"
        style={{ display: "flex", alignItems: "center" }}
      >
        <div style={{ position: "relative" }}>
          {/* Background circle */}
          <svg
            width={size}
            height={size}
            style={{ transform: "rotate(-90deg)" }}
          >
            <circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              fill="transparent"
              stroke="rgba(255, 255, 255, 0.1)"
              strokeWidth={thickness}
            />
            {/* Progress circle with neon glow effect */}
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
          {/* Percentage text with neon glow */}
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
            {normalizedPercentage}%
          </div>
        </div>
      </div>
    );
  };

  return (
    <React.Fragment>
      <h1>News Credibility Checker</h1>

      {/* Show the Search and Go button before Go is clicked */}
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

      {/* Show the Reset button after Go is clicked */}
      {goClicked && <button onClick={handleResetClick}>Reset</button>}

      {/* Loading animation and text side by side */}
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

      {/* Credibility, Circle, Authenticity, and Sources (Hidden until Go is pressed) */}
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
