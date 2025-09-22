module.exports = function(eleventyConfig) {
  // Copy everything under src/assets to _site/assets
  eleventyConfig.addPassthroughCopy({ "src/assets": "assets" });

  // Watch CSS files for changes
  eleventyConfig.addWatchTarget("./src/assets/css/");
  
  return {
    dir: {
      input: "src",
      output: "_site",
      includes: "_includes",
      layouts: "_layouts"
    },
    templateFormats: ["md", "njk", "html"],
    markdownTemplateEngine: "njk",
    htmlTemplateEngine: "njk",
    pathPrefix: process.env.ELEVENTY_PATH_PREFIX || "/bit07"

  };
};
