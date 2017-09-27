require 'fileutils'
include FileUtils

# Build Sphinx docs for Python

# Get and set release version
version = File.foreach('_config.yml').grep(/^SPARK_SKLEARN_VERSION: (.+)$/){$1}.first
version ||= 'Unknown'

puts "Moving to python directory and building sphinx."
cd("../python")
if not (ENV['SPARK_HOME'])
  raise("Python API docs cannot be generated if SPARK_HOME is not set.")
end

system({"PACKAGE_VERSION"=>version}, "./gen-doc.sh") || raise("Python doc generation failed")

puts "Moving back into home dir."
cd("../")

puts "Copying Python docs into root documentation folder (docs/)"
puts "cp -r python/doc_gen/. docs/"
cp_r("python/doc_gen/.", "docs/")

puts "rm -r python/doc_gen"
rm_r("python/doc_gen/")
