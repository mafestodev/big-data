const form = document.getElementById("identify-form");
const result = document.getElementById("result");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  result.textContent = "Identifying…";
  const response = await fetch("/api/v1/identify", {
    method: "POST",
    body: new FormData(form),
  });
  const body = await response.json();
  if (!response.ok) {
    result.textContent = body.error || "Identification failed.";
    return;
  }
  result.textContent = `${body.predicted_identity} — similarity ${body.similarity_score}`;
});

