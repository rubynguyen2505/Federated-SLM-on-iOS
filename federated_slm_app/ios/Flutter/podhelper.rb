require 'json'

def parse_flutter_config
  flutter_root = File.expand_path('..', File.dirname(__FILE__))
  generated_xcode_build_settings = File.join(flutter_root, 'flutter_export_environment.sh')
  if File.exist?(generated_xcode_build_settings)
    File.read(generated_xcode_build_settings).each_line do |line|
      if line =~ /export (\w+)="(.*)"/
        ENV[$1] = $2
      end
    end
  end
end

def flutter_install_all_ios_pods(installer)
  parse_flutter_config
  system('flutter', 'pub', 'get')
end
