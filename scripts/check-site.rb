#!/usr/bin/env ruby
# Check the generated shell without treating legacy article-body links as new routes.
require "cgi"
require "pathname"
require "uri"
require "rexml/document"
require "jekyll"

root = Pathname.new(ARGV.fetch(0, "_site")).expand_path
baseurl = ARGV.fetch(1, "").delete_suffix("/")
failures = []
check = ->(condition, message) { failures << message unless condition }

%w[index.html post-archive/index.html contact/index.html about/index.html
   collection-archive/index.html categories/index.html tags/index.html
   year-archive/index.html privacy/index.html 404.html
   assets/css/main.css assets/js/site.js assets/js/math.js feed.xml sitemap.xml].each do |path|
  check.call(root.join(path).file?, "Missing output: #{path}")
end

home = root.join("index.html").read
check.call(home.scan('class="feature-card"').size == 3, "Homepage must resolve all three featured posts")
check.call(home.include?('id="hero-title"'), "Homepage hero is missing")
check.call(home.include?('rel="canonical"'), "Canonical URL is missing")
check.call(home.include?('application/ld+json'), "Structured metadata is missing")

archive = root.join("post-archive/index.html").read
post_count = archive.scan(" data-search-item ").size
check.call(post_count.positive?, "Writing archive has no searchable posts")
check.call(archive.include?("data-search-topic"), "Topic filter is missing")

Dir.glob(root.join("**/*.html")).each do |file|
  html = File.read(file)
  next unless html.include?('class="site-header"')

  relative = Pathname.new(file).relative_path_from(root)
  check.call(html.scan(/<main\b/).size == 1, "#{relative}: expected one main landmark")
  check.call(html.include?('class="skip-link"'), "#{relative}: missing skip link")
  check.call(html.include?("#{baseurl}/assets/css/main.css"), "#{relative}: incorrect stylesheet base path")
  check.call(html.include?("#{baseurl}/assets/js/site.js"), "#{relative}: incorrect script base path")
  # Post HTML, including legacy notebook output, is deliberately not rewritten.
  shell = html.gsub(/<div class="prose"[^>]*>.*?(?=<nav class="article-tags"|<aside class="author-note")/m, "")
  shell.scan(/(?:href|src)="([^"]+)"/).flatten.each do |value|
    url = CGI.unescapeHTML(value)
    next unless url.start_with?("/") && !url.start_with?("//")

    path = URI::DEFAULT_PARSER.unescape(url.split(/[?#]/, 2).first)
    next if path.empty?
    if !baseurl.empty? && path != baseurl && !path.start_with?("#{baseurl}/")
      failures << "#{relative}: link is missing base path: #{url}"
      next
    end
    path = path.delete_prefix(baseurl).delete_prefix("/")
    target = root.join(path)
    target = target.join("index.html") if target.directory?
    check.call(target.file?, "#{relative}: broken local link: #{url}")
  end
end

%w[feed.xml sitemap.xml].each do |path|
  REXML::Document.new(root.join(path).read)
end

%w[README.md Gemfile Gemfile.lock _config.yml.save scripts _notebooks _tmp _trash].each do |path|
  check.call(!root.join(path).exist?, "Source-only file published: #{path}")
end

converter = Jekyll::Converters::Markdown.new(Jekyll.configuration("quiet" => true))
{
  "python" => "def square(x):\n    return x ** 2",
  "javascript" => "const square = (x) => x ** 2;",
  "sql" => "SELECT score FROM experiments;",
  "bash" => "echo \"$HOME\""
}.each do |language, code|
  rendered = converter.convert("```#{language}\n#{code}\n```\n")
  check.call(rendered.include?("language-#{language}") && rendered.include?("<span class="),
             "Markdown no longer highlights #{language} fences")
end

details = converter.convert("<details markdown=\"1\">\n<summary>Code</summary>\n\n```python\nprint(42)\n```\n\n</details>\n")
check.call(details.include?("<details>") && details.include?('class="language-python'),
           "Native details must support highlighted Markdown")
math = converter.convert("Inline $x^2$.\n\n$$\\sum_{i=1}^n i$$\n")
check.call(math.include?("$x^2$") && math.include?("\\[\\sum_{i=1}^n i\\]"),
           "LaTeX delimiters must survive Markdown conversion")

abort failures.uniq.join("\n") unless failures.empty?
puts "Site checks passed (#{post_count} published articles)."
