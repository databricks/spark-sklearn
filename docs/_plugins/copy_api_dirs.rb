#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

require 'fileutils'
include FileUtils

if not (ENV['SKIP_API'] == '1')
    # Build Sphinx docs for Python

    # Get and set release version
    version = File.foreach('_config.yml').grep(/^SPARK_SKLEARN_VERSION: (.+)$/){$1}.first
    version ||= 'Unknown'

    puts "Moving to python directory and building sphinx."
    cd("../python")
    if not (ENV['SPARK_HOME'])
      raise("Python API docs cannot be generated if SPARK_HOME is not set.")
    end
    # system({"PACKAGE_VERSION"=>version}, "make clean") || raise("Python doc clean failed")
    system({"PACKAGE_VERSION"=>version}, "./gen-doc.sh") || raise("Python doc generation failed")

    puts "Moving back into home dir."
    cd("../")

    puts "Copying Python docs into root documentation folder (docs/)"
    puts "cp -r python/doc_gen/. docs/"
    cp_r("python/doc_gen/.", "docs/")

    puts "rm -r python/doc_gen"
    rm_r("python/doc_gen/")
end
