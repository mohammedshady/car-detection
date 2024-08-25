import FileUpload from "./FileUpload";
import "./App.css";

function App() {
  const handleClearButton = () => {
    window.location.reload();
  };
  return (
    <>
      <button onClick={handleClearButton}>Clear</button>
      <FileUpload />
    </>
  );
}

export default App;
