// Find the section with ID "article-content"
const articleContent = document.getElementsByClassName('article-content')[0];

// Check if the section exists
if (articleContent) {
  // Find all <a> tags with the class "footnote-ref" within the section
  const footnoteRefs = articleContent.querySelectorAll('a.footnote-ref');

  // Loop through each <a> tag and modify the text content
  footnoteRefs.forEach((footnoteRef) => {
    const text = footnoteRef.textContent;
    footnoteRef.textContent = `[${text}]`;
  });
}