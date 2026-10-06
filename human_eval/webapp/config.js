// Runtime config for the human-validation SPA.
// Edit API_BASE_URL after `sam deploy` prints the ApiUrl output, then
// re-sync webapp/ to the site bucket. Leave it empty to run fully offline
// (labels are kept in localStorage and can be exported as JSON on the
// done screen).
window.HALLUVISION_CONFIG = {
  // e.g. "https://abc123.execute-api.ap-northeast-2.amazonaws.com"
  API_BASE_URL: "",

  // The evaluators. `id` is what the backend stores (a / b / c / d …) and must
  // match the ids passed to assign_items.py. Rename c/d (or add more) to your
  // actual 3rd/4th annotators; an "Other" option is always added by the app for
  // ad-hoc testers (name + affiliation, shown all items, excluded from κ).
  ANNOTATORS: [
    { id: "a", name: "Annotator A", affil: "" },
    { id: "b", name: "Annotator B", affil: "" },
    { id: "c", name: "Annotator C", affil: "" },
    { id: "d", name: "Annotator D", affil: "" },
    { id: "e", name: "Annotator E", affil: "" },
    { id: "f", name: "Annotator F", affil: "" },
    { id: "g", name: "Annotator G", affil: "" },
    { id: "h", name: "Annotator H", affil: "" },
    { id: "i", name: "Annotator I", affil: "" },
    { id: "j", name: "Annotator J", affil: "" },
  ],
};
